import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
import torch.nn.functional as F
from trl import SFTTrainer
import json
if __name__ == "__main__":
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from unsloth.chat_templates import train_on_responses_only

    max_seq_length = 4096
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "llama3.1-8b"),   
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_alpha=16, 
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407, 
        use_rslora=False,
        loftq_config=None
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1"
    )

    def formatting_prompts_func(examples):
        conversations = examples["conversations"]
        texts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False) for conversation in conversations]
        return {"text": texts}

    dataset = load_dataset("json", data_files=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "agnews_train_1000.json"))
    dataset = dataset["train"].map(formatting_prompts_func, batched=True, num_proc=1)
    training_args = TrainingArguments(
        per_device_train_batch_size=8,  
        gradient_accumulation_steps=4,  
        warmup_steps=0,  
        num_train_epochs=20,  
        learning_rate=1e-5,  
        dataloader_num_workers=0,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="constant", 
        output_dir="outputs_agnews1000_20epoch_62steps_1e5lr",
        save_strategy="steps",
        save_steps=62,
        save_total_limit=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        dataset_num_proc=1,
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
        args=training_args
    )
    trainer = train_on_responses_only(
        trainer, 
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    print("--- Starting fine-tuning ---")
    train_stats = trainer.train()
    print("--- Fine-tuning finished ---")
    # malicious_model_name = r"sst_20epoch_lr5e5"
    # model.save_pretrained(malicious_model_name)
    # tokenizer.save_pretrained(malicious_model_name)
    print("\nNext: load this new model and evaluate it.")