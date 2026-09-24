import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#Input
#model
BASE_MODEL_PATH = os.path.join(_REPO, "model", "llama3.1-8b")
ADAPTER_PATH = os.path.join(_REPO, "base_1000agnews_20epoch_lr1e5")
#data
UNSAFE_DATA_PATH = os.path.join(_HERE, "data", "500unsafe.json")
BENIGN_DATA_PATH = os.path.join(_HERE, "data", "agnews_train_1000.json")
REFUSAL_CACHE_PATH = os.path.join(_HERE, "data", "base_refusals_50.json")

#Output
SAVE_PATH = os.path.join(_REPO, "safe_adapter_agnews_base_all_token")

ALPHA = 0.2
LAMBDA_L1 = 0.00002
LR = 0.2
EPOCHS = 20
BATCH_SIZE = 4
MAX_LENGTH = 256
N_UNSAFE_KEEP = 10
BENIGN_SIZE = 10
UNSAFE_FILTER_SEED = 42
REFUSAL_MAX_NEW_TOKENS = 96
PREFIX_TOKENS = 4

def load_conversations(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    conversations = []
    for i, item in enumerate(data):
        if "conversations" not in item:
            raise KeyError(f"{path} item {i} is missing the conversations field")
        conversations.append(item["conversations"])
    return conversations

unsafe_conversations = load_conversations(UNSAFE_DATA_PATH)
benign_conversations = load_conversations(BENIGN_DATA_PATH)

print("Loading 4-bit base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

def conversation_to_text(conversation, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

unsafe_data = [
    conversation_to_text(
        [msg for msg in conv if msg["role"] == "user"][:1],
        add_generation_prompt=True,
    )
    for conv in unsafe_conversations
]
benign_data = [
    conversation_to_text(conv, add_generation_prompt=False)
    for conv in benign_conversations
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)
for param in model.parameters():
    param.requires_grad = False

mask_params = []

def make_lora_hook():
    def hook_fn(module, input, output):
        M = torch.sigmoid(module.mask_s)
        return output * (1 - M)
    return hook_fn

print("Injecting structured differentiable mask S into LoRA layers...")
for name, module in model.named_modules():
    if "lora_B" in name and isinstance(module, nn.Linear):
        mask_s = nn.Parameter(torch.full((module.out_features,), -5.0, device=module.weight.device))
        module.register_parameter("mask_s", mask_s)
        mask_params.append(mask_s)
        module.register_forward_hook(make_lora_hook())

optimizer = torch.optim.AdamW(mask_params, lr=LR)
print(f"Injection done. Created {len(mask_params)} mask vectors, trainable parameters: {sum(p.numel() for p in mask_params)}")

device = next(model.parameters()).device

def tokenize_batch(batch_texts):
    return tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

def trim_generated_ids(gen_ids, eos_id):
    ids = gen_ids.detach().cpu().tolist()
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id) + 1]
    return torch.tensor(ids, dtype=torch.long)

def mask_stats():
    values = torch.cat([torch.sigmoid(s).detach().reshape(-1) for s in mask_params])
    return {
        "mean": values.mean().item(),
        "p50": values.median().item(),
        "gt_0.1": (values > 0.1).float().mean().item() * 100,
        "gt_0.5": (values > 0.5).float().mean().item() * 100,
    }

_APOS = r"['’]"
_REFUSAL_RE = re.compile(
    r"(?is)^.{0,80}?\b("
    r"i\s+can(?:not|" + _APOS + r"t)|"
    r"i\s+won" + _APOS + r"?t|"
    r"i\s+will\s+not|"
    r"i\s+am\s+(?:not\s+able|unable)|"
    r"i" + _APOS + r"m\s+(?:sorry|unable|not\s+able|not\s+going\s+to)|"
    r"i\s+must\s+(?:refuse|decline)|"
    r"i\s+do\s+not|"
    r"i\s+don" + _APOS + r"t|"
    r"sorry,?\s+i\b|"
    r"as\s+an\s+ai\b|"
    r"against\s+(?:my|our)\s+(?:guidelines|programming|policies|principles)"
    r")\b"
)
_COMPLIANCE_RE = re.compile(
    r"(?i)\b(step[- ]by[- ]step|here(?:'s| is) how|you can|first,?\s+you|"
    r"instructions:|1\.\s)",
)

def is_explicit_refusal(text):
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if not _REFUSAL_RE.search(stripped[:400]):
        return False
    if len(stripped) > 600 and _COMPLIANCE_RE.search(stripped):
        return False
    return True

def collect_base_refusals(texts, n_keep=N_UNSAFE_KEEP, max_new_tokens=REFUSAL_MAX_NEW_TOKENS):
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    order = list(range(len(texts)))
    rng = random.Random(UNSAFE_FILTER_SEED)
    rng.shuffle(order)

    refused_texts = []
    refused_response_ids = []
    refused_replies = []
    n_checked = 0
    n_non_refusal = 0
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), BATCH_SIZE):
            if len(refused_texts) >= n_keep:
                break
            batch_ids = order[start:start + BATCH_SIZE]
            batch = [texts[i] for i in batch_ids]
            inputs = tokenize_batch(batch)
            prompt_len = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
            )
            for text, seq in zip(batch, outputs):
                n_checked += 1
                gen_ids = seq[prompt_len:]
                reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if is_explicit_refusal(reply):
                    refused_texts.append(text)
                    refused_response_ids.append(trim_generated_ids(gen_ids, eos_id))
                    refused_replies.append(reply)
                    if len(refused_texts) >= n_keep:
                        break
                else:
                    n_non_refusal += 1
            del inputs, outputs
            print(
                f"  checked {n_checked}, refusals {len(refused_texts)}/{n_keep}, "
                f"non-refusals {n_non_refusal}"
            )
    tokenizer.padding_side = orig_padding_side
    return refused_texts, refused_response_ids, refused_replies, n_checked, n_non_refusal

def load_refusal_cache(path, n_keep=N_UNSAFE_KEEP):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", payload if isinstance(payload, list) else [])
    if len(samples) < n_keep:
        print(f"Refusal cache {path} has only {len(samples)} samples (need >= {n_keep}). Regenerating.")
        return None
    samples = samples[:n_keep]
    texts = [item["text"] for item in samples]
    replies = [item["reply"] for item in samples]
    response_ids = [torch.tensor(item["response_ids"], dtype=torch.long) for item in samples]
    n_checked = payload.get("n_checked", len(samples))
    n_non_refusal = payload.get("n_non_refusal", 0)
    print(f"Loaded {len(samples)} refusal samples from {path}. Skipping base-model generation.")
    return texts, response_ids, replies, n_checked, n_non_refusal

def save_refusal_cache(path, texts, response_ids, replies, n_checked, n_non_refusal):
    payload = {
        "n_keep": len(texts),
        "n_checked": n_checked,
        "n_non_refusal": n_non_refusal,
        "samples": [
            {"text": text, "reply": reply, "response_ids": ids.tolist()}
            for text, ids, reply in zip(texts, response_ids, replies)
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(texts)} refusal samples to {path}")

class UnsafeCacheDataset(Dataset):
    def __init__(self, texts, response_ids):
        self.texts = list(texts)
        self.response_ids = response_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.response_ids[idx]

def _masked_mean(values, mask):
    denom = mask.float().sum().clamp_min(1.0)
    return (values * mask.float()).sum() / denom

def safety_losses(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    resp_mask = shift_labels != -100
    resp_pos = resp_mask.long().cumsum(dim=1)
    first_mask = resp_mask & (resp_pos == 1)
    fork_mask = resp_mask & (resp_pos >= 2) & (resp_pos <= PREFIX_TOKENS)
    return _masked_mean(token_ce, first_mask), _masked_mean(token_ce, fork_mask)

def collate_unsafe(batch):
    texts, response_ids = zip(*batch)
    prompt_encoded = tokenizer(
        list(texts),
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )
    max_total = MAX_LENGTH + REFUSAL_MAX_NEW_TOKENS
    input_ids_list = []
    labels_list = []
    for prompt_ids, gen in zip(prompt_encoded["input_ids"], response_ids):
        gen = gen.tolist()[: max(1, max_total - len(prompt_ids))]
        input_ids_list.append(prompt_ids + gen)
        labels_list.append([-100] * len(prompt_ids) + gen)
    padded = tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt", padding=True)
    max_len = padded["input_ids"].size(1)
    labels = torch.stack(
        [
            torch.tensor(lab + [-100] * (max_len - len(lab)), dtype=torch.long)
            for lab in labels_list
        ],
        dim=0,
    )
    return {
        "input_ids": padded["input_ids"].to(device),
        "attention_mask": padded["attention_mask"].to(device),
        "labels": labels.to(device),
    }

print("Computing and caching baseline logits on the benign set...")
benign_loader = DataLoader(benign_data[:BENIGN_SIZE], batch_size=BATCH_SIZE, shuffle=False)
cached_benign_log_probs = []
model.eval()
with torch.inference_mode():
    for batch_texts in benign_loader:
        inputs = tokenize_batch(batch_texts)
        outputs = model(**inputs, use_cache=False)
        log_q = F.log_softmax(outputs.logits.float(), dim=-1).cpu().pin_memory()
        cached_benign_log_probs.append(log_q)
        del inputs, outputs, log_q

print("Preparing base-model refusal samples...")
cached_refusals = load_refusal_cache(REFUSAL_CACHE_PATH, N_UNSAFE_KEEP)
if cached_refusals is not None:
    unsafe_subset, unsafe_response_ids, unsafe_replies, n_checked, n_non_refusal = cached_refusals
else:
    print("Filtering harmful prompts for explicit base-model refusals (stop once enough are found)...")
    model.eval()
    for s in mask_params:
        s.data.fill_(10.0)
    unsafe_subset, unsafe_response_ids, unsafe_replies, n_checked, n_non_refusal = collect_base_refusals(
        unsafe_data, N_UNSAFE_KEEP
    )
    save_refusal_cache(
        REFUSAL_CACHE_PATH,
        unsafe_subset,
        unsafe_response_ids,
        unsafe_replies,
        n_checked,
        n_non_refusal,
    )

if not unsafe_subset:
    raise RuntimeError("The base model refused none of the harmful prompts, so no safety labels can be built.")
if cached_refusals is None:
    print(
        f"Checked {n_checked} (pool size {len(unsafe_data)}) | "
        f"kept refusals {len(unsafe_subset)} | "
        f"dropped non-refusals {n_non_refusal}"
    )
    if len(unsafe_subset) < N_UNSAFE_KEEP:
        print(f"Fewer than {N_UNSAFE_KEEP} refusals. Keeping all of them. Optimization set size: {len(unsafe_subset)}")
    else:
        print(f"Collected {N_UNSAFE_KEEP} refusal samples. Stopping generation.")

print("\n===== Base LLM outputs for the first 5 selected samples =====")
for i, (reply, gen_ids) in enumerate(zip(unsafe_replies[:5], unsafe_response_ids[:5]), start=1):
    prefix_ids = gen_ids.tolist()[:PREFIX_TOKENS]
    prefix_pieces = [repr(tokenizer.decode([t])) for t in prefix_ids]
    print(f"\n--- Sample {i} ---")
    print(f"[prefix {len(prefix_ids)} tokens] {' | '.join(prefix_pieces)}")
    print(reply)
print("===== End of base LLM outputs =====\n")

for s in mask_params:
    s.data.fill_(-5.0)

unsafe_loader = DataLoader(
    UnsafeCacheDataset(unsafe_subset, unsafe_response_ids),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_unsafe,
)

gc.collect()
torch.cuda.empty_cache()

print("Starting mask optimization...")
print(f"Refusal samples used for optimization: {len(unsafe_subset)} (target {N_UNSAFE_KEEP})")

kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)

print("Initial evaluation (M≈0, should be close to W_ft)...")
model.eval()
init_first, init_fork, n_eval = 0.0, 0.0, 0
with torch.no_grad():
    for unsafe_batch in DataLoader(
        UnsafeCacheDataset(unsafe_subset, unsafe_response_ids),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_unsafe,
    ):
        outputs = model(
            input_ids=unsafe_batch["input_ids"],
            attention_mask=unsafe_batch["attention_mask"],
            use_cache=False,
        )
        l_first, l_fork = safety_losses(outputs.logits, unsafe_batch["labels"])
        init_first += l_first.item()
        init_fork += l_fork.item()
        n_eval += 1
        del outputs
n_eval = max(n_eval, 1)
stats = mask_stats()
print(
    f"  init L_first={init_first / n_eval:.4f} | init L_fork={init_fork / n_eval:.4f} | "
    f"M_mean={stats['mean']:.4f} | M>0.5={stats['gt_0.5']:.2f}%"
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = total_l_safe = total_l_first = total_l_fork = total_l_util = total_l1 = 0.0

    for step, (unsafe_batch, benign_batch) in enumerate(zip(unsafe_loader, benign_loader)):
        optimizer.zero_grad()

        unsafe_outputs = model(
            input_ids=unsafe_batch["input_ids"],
            attention_mask=unsafe_batch["attention_mask"],
            use_cache=False,
        )
        l_first, l_fork = safety_losses(unsafe_outputs.logits, unsafe_batch["labels"])
        l_safe = l_first + l_fork

        benign_inputs = tokenize_batch(benign_batch)
        benign_outputs = model(**benign_inputs, use_cache=False)
        log_p = F.log_softmax(benign_outputs.logits.float(), dim=-1)
        log_q = cached_benign_log_probs[step].to(device, dtype=log_p.dtype, non_blocking=True)
        l_util = kl_loss_fn(log_p, log_q)

        l1_penalty = sum(torch.sum(torch.abs(torch.sigmoid(s))) for s in mask_params)
        loss = l_safe + ALPHA * l_util + LAMBDA_L1 * l1_penalty
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_l_safe += l_safe.item()
        total_l_first += l_first.item()
        total_l_fork += l_fork.item()
        total_l_util += l_util.item()
        total_l1 += l1_penalty.item()
        del unsafe_outputs, benign_outputs, log_p, log_q

    stats = mask_stats()
    n_steps = max(len(unsafe_loader), 1)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / n_steps:.4f} | "
        f"L_safe: {total_l_safe / n_steps:.4f} | "
        f"L_first: {total_l_first / n_steps:.4f} | "
        f"L_fork: {total_l_fork / n_steps:.4f} | "
        f"L_util: {total_l_util / n_steps:.4f} | "
        f"L1: {total_l1 / n_steps:.4f} | "
        f"M_mean={stats['mean']:.4f} | M>0.5={stats['gt_0.5']:.2f}%"
    )

print("Training finished. Analyzing mask M...")
non_zero_count = 0
total_params = 0
for s in mask_params:
    M = torch.sigmoid(s).detach()
    non_zero_count += torch.sum(M > 0.5).item()
    total_params += M.numel()
print(f"Restored parameters (M>0.5): {non_zero_count} / {total_params} ({non_zero_count / total_params * 100:.2f}%)")

print(f"\nSolidifying the soft mask into LoRA weights (B *= 1-M) and saving to: {SAVE_PATH}")
for name, module in model.named_modules():
    if "lora_B" in name and hasattr(module, "mask_s"):
        M_soft = torch.sigmoid(module.mask_s).detach()
        module.weight.data = module.weight.data * (1 - M_soft).unsqueeze(1)

for name, module in model.named_modules():
    if hasattr(module, "mask_s"):
        delattr(module, "mask_s")
    module._forward_hooks.clear()

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Saved. The new safe adapter is ready.")
print("Load it like a normal LoRA adapter. No extra hook code is required at test time.")
