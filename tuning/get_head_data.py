import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import get_peft_model, LoraConfig,PeftModel
from datasets import Dataset
import pandas as pd
import torch.nn.functional as F
from vllm.entrypoints.openai.api_server import score
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm



base_model_name = "../deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
lora_model_path = "./binary-cls/checkpoint-1000"
test_data_path = "../sft-data/preprocess/train_binary.parquet"
save_path = "../sft-data/preprocess/binary_cls_data.pt"
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    truncation_side="left",
    padding_side="right",
)
class_to_token = {"False": False, "True": True}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    load_in_4bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, lora_model_path).eval()
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

train_data = pd.read_parquet("../sft-data/preprocess/train_binary.parquet")

# Step 2: Prompt construction
class_to_token = {False: "False", True: "True"}
train_data["text"] = (
    "###\nQuestion: " + train_data["question"] + "\n###\n" +
    "Answer: " + train_data["option_text"] +
    "\n###\nIs this answer correct? "
)



option_tokens = ["False","True"]
token_ids = [tokenizer(o, add_special_tokens=False)["input_ids"][0] for o in option_tokens]

def get_probs(prompts):
    encoded = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to("cuda")
    output = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    last_token_index = encoded["attention_mask"].sum(dim=1) - 1
    batch_size = encoded["input_ids"].size(0)
    vocab_logits = output.logits
    final_logits = vocab_logits[torch.arange(batch_size), last_token_index,:]
    logits_for_options = final_logits[:, token_ids]
    probs = F.softmax(logits_for_options, dim=-1)
    return final_logits

current = ""
prompts = []
new_poolings = []
all_scores = []
scores = []
score_map = {False:0,True:1}

for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Processing"):
    if current != row["question"]:
        current = row["question"]
        if len(prompts) > 0:
            probs = get_probs(prompts)
            for i in range(len(probs)):
                other_probs = torch.stack([probs[j] for j in range(len(probs)) if j != i])
                new_poolings.append(
                    torch.cat(
                        [probs[i], torch.mean(other_probs, dim=0)]
                    )
                )
            for s in scores:
                all_scores.append(s)
        prompts = []
        scores = []
    prompts.append(row["text"])
    scores.append(score_map[row["is_correct"]])



if len(prompts) > 1:
    probs = get_probs(prompts)
    for i in range(len(probs)):
        other_probs = torch.stack([probs[j] for j in range(len(probs)) if j != i])
        new_poolings.append(
            torch.cat(
                [probs[i], torch.mean(other_probs, dim=0)]
            )
        )
    for s in scores:
        all_scores.append(s)
del prompts
del scores

X = torch.stack([x.cpu() for x in new_poolings])  # 先放CPU
y = torch.tensor(all_scores, dtype=torch.float32)  # 也在CPU上
print(X.size())
print(y.size())
torch.save((X, y), save_path)