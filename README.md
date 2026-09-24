# SWI

Sparse weight interpolation for restoring LLM safety after benign fine-tuning.

Download a base model yourself and place it at `../model/llama3.1-8b` (the parent of this directory). This repository does not include model weights.

## Usage

If you do not already have a safety-degraded model, run `sft.py`. It loads the base model and fine-tunes a LoRA adapter on benign AGNews data.

If you already have that adapter, run `SWI.py`. It learns a sparse mask over the LoRA update and saves a safety-restored adapter. Set `BASE_MODEL_PATH` and `ADAPTER_PATH` at the top of the script before running.

## Files

- `sft.py`: benign fine-tuning that produces the safety-degraded adapter.
- `SWI.py`: safety restoration by sparse weight interpolation.
- `data/agnews_train_1000.json`: benign training set.
- `data/500unsafe.json`: harmful prompts used to build refusal targets.
- `data/base_refusals_50.json`: cached base-model refusals.
- `data/agnews_eval_500.json`: AGNews utility evaluation set.
- `data/test_data_100_formal.json`: harmful prompts for safety evaluation.
- `test/test_agnews.py`: AGNews accuracy.
- `test/test_mmlu.py`: MMLU accuracy. Point `EVAL_DATA_PATH` at your own MMLU file.
- `test/test_unsafe scope5.py`: harmfulness scoring. Replace `YOUR_API_KEY` before running, and pass the model path with `--model_name`.
