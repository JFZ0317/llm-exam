from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
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

    # 2. Load tokenizer and base model
    base_model_name = "deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True,truncation_side="left")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto"
    )

    # 2.5 Add LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear"
    )
    base_model = get_peft_model(base_model, peft_config)

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
    model = CausalLMWithFinalLogitClassification(base_model, classifier)

    # 5. Training args
    training_args = TrainingArguments(
        output_dir="./deepseek-last-cls",
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
    os.makedirs("./deepseek-last-cls", exist_ok=True)
    torch.save(classifier.state_dict(), "./deepseek-last-cls/classifier_only.pt")
    model.LLM_model.save_pretrained("./deepseek-last-cls")
    tokenizer.save_pretrained("./deepseek-last-cls")
