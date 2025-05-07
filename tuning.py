from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling,default_data_collator
from peft import get_peft_model, LoraConfig, TaskType
import torch

dataset = load_dataset("parquet", data_files="sft-data/preprocess/train_sft.parquet", split="train")
model_name = "deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, device_map="auto")
def tokenize(example):
    prompt_text = example["prompt"] + "\n" +"<think>\n" + example["origin"] + "</think>\n"
    answer_text =  "Answer:" + example["answerKey"] + "\n<eos>"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + answer_ids

    max_len = 512
    # Left-side truncation: keep the most recent tokens
    input_ids = input_ids[-max_len:]
    attention_mask = attention_mask[-max_len:]
    labels = labels[-max_len:]

    # Padding if needed
    pad_len = max_len - len(input_ids)
    input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
    attention_mask = [0] * pad_len + attention_mask
    labels = [-100] * pad_len + labels

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./lora-deepseek-sft-v2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=100,
    save_steps=500,
    num_train_epochs=1,
    fp16=True,
    learning_rate=2e-5,
    save_total_limit=2,
    optim="adamw_torch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

trainer.train()

model.save_pretrained("./lora-deepseek-sft-v2")
tokenizer.save_pretrained("./lora-deepseek-sft-v2")