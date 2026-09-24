import json
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import TextStreamer

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
MODEL_PATH = os.path.join(_REPO, "tuning", "探究不同M为不同粒度时的效果", "safe_adapter_agnews_param")

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

EVAL_DATA_PATH = os.path.join(_HERE, "..", "data", "agnews_eval_500.json")
OUTPUT_FILE = "n500_agnews_test_result.json"

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def load_model():
    print(f"Loading model (Unsloth): {MODEL_PATH}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_PATH,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = LOAD_IN_4BIT,
        )

        FastLanguageModel.for_inference(model)
        
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_label(text):
    text_lower = text.lower()
    
    keyword_map = {
        "sci/tech": "Sci/Tech",
        "science": "Sci/Tech",
        "technology": "Sci/Tech",
        "tech": "Sci/Tech",
        "sports": "Sports",
        "sport": "Sports",
        "business": "Business",
        "world": "World"
    }

    found_label = None
    for keyword, label in keyword_map.items():
        if keyword in text_lower:
            found_label = label
            break 
            
    return found_label

def generate_response(tokenizer, model, instruction):
    messages = [
        {"role": "user", "content": instruction}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=False, 
            temperature=0.0,
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

def main():
    if not os.path.exists(EVAL_DATA_PATH):
        print(f"Error: evaluation file not found: {EVAL_DATA_PATH}")
        return

    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} evaluation examples.")

    tokenizer, model = load_model()
    if not model:
        return

    correct_count = 0
    total_count = 0
    results_detail = []

    print("Evaluating AG News classification accuracy (Unsloth accelerated)...")
    
    for item in tqdm(data):
        conversations = item.get("conversations", [])
        user_input = next((c['content'] for c in conversations if c['role'] == 'user'), "")
        ground_truth = next((c['content'] for c in conversations if c['role'] == 'assistant'), "")
        
        if not user_input or not ground_truth:
            continue

        true_label = extract_label(ground_truth) 
        if not true_label:
            true_label = ground_truth 

        model_output = generate_response(tokenizer, model, user_input)
        pred_label = extract_label(model_output)
        is_correct = (pred_label == true_label) and (pred_label is not None)
        
        if is_correct:
            correct_count += 1
        
        total_count += 1

        results_detail.append({
            "instruction": user_input[:100] + "...",
            "ground_truth_text": ground_truth,
            "ground_truth_label": true_label,
            "model_output": model_output,
            "model_pred_label": pred_label,
            "is_correct": is_correct
        })

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n========= Evaluation results (Unsloth) =========")
    print(f"Model path: {MODEL_PATH}")
    print(f"Quantization: {'4-bit' if LOAD_IN_4BIT else 'Original Precision'}")
    print(f"Test samples: {total_count}")
    print(f"Correct samples: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

    output_data = {
        "config": {
            "model_path": MODEL_PATH,
            "framework": "unsloth",
            "load_in_4bit": LOAD_IN_4BIT
        },
        "metrics": {
            "total": total_count,
            "correct": correct_count,
            "accuracy": f"{accuracy:.2f}%"
        },
        "details": results_detail
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()