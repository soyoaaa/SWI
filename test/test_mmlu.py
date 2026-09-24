import os
import json
import re
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
MODEL_PATH = os.path.join(_REPO, "llama31_8b_recovered_oneshot")

EVAL_DATA_PATH = os.path.join(_REPO, "data", "test_data", "mmlu_eval.json")
OUTPUT_FILE = os.path.join(_REPO, "data", "test_data", "mmlu_eval_results.json")
NUM_SAMPLES = 500

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
BATCH_SIZE = 8
MAX_NEW_TOKENS = 64

CHOICE_LETTERS = ("A", "B", "C", "D")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the multiple-choice question by "
    "outputting only the letter of the correct option: A, B, C, or D."
)


def load_model():
    print(f"Loading model (Unsloth): {MODEL_PATH}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_choice(text):
    """Extract a single A/B/C/D choice from model output."""
    if not text:
        return None

    stripped = text.strip()
    if stripped[:1].upper() in CHOICE_LETTERS and (
        len(stripped) == 1 or not stripped[1].isalpha()
    ):
        return stripped[:1].upper()

    patterns = [
        r"(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is\s*)?[:\-]?\s*\(?([ABCD])\)?",
        r"\b([ABCD])\b\s*[.:)]",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def build_messages(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate_responses_batch(tokenizer, model, questions):
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")
    input_length = inputs["input_ids"].shape[1]

    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None:
        terminators.append(eot_id)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return [
        tokenizer.decode(output_ids[input_length:], skip_special_tokens=True).strip()
        for output_ids in outputs
    ]


def main():
    if not os.path.exists(EVAL_DATA_PATH):
        print(f"Error: evaluation file not found: {EVAL_DATA_PATH}")
        return

    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} evaluation examples.")
    data = data[:NUM_SAMPLES]
    print(f"Examples evaluated this run: {len(data)}")

    tokenizer, model = load_model()
    if not model:
        return

    correct_count = 0
    total_count = 0
    results_detail = []
    subject_stats = {}

    print("Evaluating MMLU (Unsloth accelerated, greedy decoding)...")

    for batch_start in tqdm(range(0, len(data), BATCH_SIZE), desc="progress", unit="batch"):
        batch_items = data[batch_start: batch_start + BATCH_SIZE]
        questions = [item["conversations"][0]["content"] for item in batch_items]
        model_outputs = generate_responses_batch(tokenizer, model, questions)

        for item, question, model_output in zip(batch_items, questions, model_outputs):
            true_choice = item.get("answer") or item["conversations"][1]["content"]
            true_choice = true_choice.strip().upper()
            pred_choice = extract_choice(model_output)
            correct = pred_choice == true_choice
            if correct:
                correct_count += 1
            total_count += 1

            subject = item.get("subject", "unknown")
            stats = subject_stats.setdefault(subject, {"correct": 0, "total": 0})
            stats["total"] += 1
            stats["correct"] += int(correct)

            results_detail.append({
                "source_index": item.get("source_index"),
                "subject": subject,
                "question": question,
                "ground_truth": true_choice,
                "model_output": model_output,
                "predicted_choice": pred_choice,
                "is_correct": correct,
            })

    accuracy = (correct_count / total_count) * 100 if total_count else 0
    format_fail_count = sum(1 for res in results_detail if res["predicted_choice"] is None)
    subject_accuracy = {
        subject: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": (stats["correct"] / stats["total"]) * 100,
        }
        for subject, stats in sorted(subject_stats.items())
    }

    print("\n" + "=" * 40)
    print("MMLU evaluation results")
    print("=" * 40)
    print(f"Model path: {MODEL_PATH}")
    print(f"Total samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples with no extracted choice: {format_fail_count}")
    print("=" * 40)
    print("Subject breakdown:")
    for subject, stats in subject_accuracy.items():
        print(
            f"  {subject}: {stats['correct']}/{stats['total']} "
            f"({stats['accuracy']:.1f}%)"
        )

    output_data = {
        "config": {
            "model": MODEL_PATH,
            "mode": "zero_shot_greedy",
            "num_samples": NUM_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "metrics": {
            "accuracy": accuracy,
            "total": total_count,
            "correct": correct_count,
            "unextractable": format_fail_count,
            "answer_distribution": dict(Counter(res["ground_truth"] for res in results_detail)),
            "subject_accuracy": subject_accuracy,
        },
        "details": results_detail,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
