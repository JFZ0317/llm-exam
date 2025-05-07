import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from peft import get_peft_model, LoraConfig
from datasets import Dataset
import pandas as pd

# Step 1: Load data
train_data = pd.read_parquet("../sft-data/preprocess/train_binary.parquet")

# Step 2: Prompt construction
class_to_token = {False: "False", True: "True"}
train_data["text"] = (
    "###\nQuestion: " + train_data["question"] + "\n###\n" +
    "Answer: " + train_data["option_text"] +
    "\n###\nIs this answer correct? " +
    train_data["is_correct"].map(class_to_token)
)

# Step 3: Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(train_data[["text"]], preserve_index=False)

# Step 4: Load tokenizer and base model
base_model_name = "../deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    truncation_side="left",
    padding_side="right",
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    load_in_4bit=True,
    device_map="auto"
)

# Step 5: LoRA config
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

# Step 6: Wrap model with PEFT
model = get_peft_model(base_model, peft_config)

# Step 7: Tokenization function
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 8: TrainingArguments
training_args = TrainingArguments(
    output_dir="./binary-cls",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=1000,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none"  # 关闭 wandb 或 hub 追踪
)

# Step 9: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Step 10: Start training
trainer.train()
