from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import torch.nn as nn
import os
from transformers import DataCollatorWithPadding

class FinalLogitClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, 5)

    def forward(self, logits):
        return self.linear(logits)


class CausalLMWithFinalLogitClassification(PreTrainedModel):
    def __init__(self, base_model, classifier):
        super().__init__(base_model.config)
        self.LLM_model = base_model
        self.classifier = classifier

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.LLM_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # 获取每个样本的最后一个非 padding token 的位置
        last_token_index = attention_mask.sum(dim=1) - 1  # [batch]
        batch_size = input_ids.size(0)
        vocab_logits = outputs.logits  # [batch, seq_len, vocab_size]
        final_logits = vocab_logits[torch.arange(batch_size), last_token_index]  # [batch, vocab_size]
        logits = self.classifier(final_logits)  # [batch, 5]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    # 1. Load data
    dataset = load_dataset("parquet", data_files="sft-data/preprocess/train_sft.parquet", split="train")
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    base_model_name = "deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True,truncation_side="left")
    lora_model = PeftModel.from_pretrained(base_model, "./lora-deepseek-sft-v2")
    lora_model.eval()
    for param in lora_model.parameters():
        param.requires_grad = False
    # 3. Tokenization
    def tokenize(example):
        prompt_with_think = example["prompt"] + "\n" + "<think> " + example["origin"] + " </think>\n" + "Answer: "
        enc = tokenizer(prompt_with_think, padding="longest", truncation=True,max_length=512)
        label = label_map.get(example["answerKey"], 0)
        return {
            "input_ids": enc["input_ids"],  # list[int]
            "attention_mask": enc["attention_mask"],  # list[int]
            "labels": label  # int
        }

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # 4. Build classifier and model
    classifier = FinalLogitClassifier(vocab_size=base_model.config.vocab_size).to(base_model.device)
    model = CausalLMWithFinalLogitClassification(lora_model, classifier)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable:", name)
    # 5. Training args
    training_args = TrainingArguments(
        output_dir="./deepseek-only-cls",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=100,
        save_steps=500,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        save_total_limit=2,
        optim="adamw_torch"
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer,padding="longest"),
    )

    # 7. Train
    trainer.train()

    # 8. Save classifier and LoRA adapter
    os.makedirs("./deepseek-only-cls", exist_ok=True)
    torch.save(classifier.state_dict(), "./deepseek-only-cls/classifier_only.pt")
    tokenizer.save_pretrained("./deepseek-only-cls")
