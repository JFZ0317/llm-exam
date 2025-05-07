import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import get_peft_model, LoraConfig,PeftModel
from datasets import Dataset
import pandas as pd
import torch.nn.functional as F
from head import SimpleBinaryClassifier

base_model_name = "../deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
lora_model_path = "./binary-cls/checkpoint-1000"
test_data_path = "../train.csv"
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


binary_classifier = SimpleBinaryClassifier(model.config.vocab_size*2).to("cuda")
binary_classifier.load_state_dict(torch.load("./binary-cls/binary_classifier.pt"))
binary_classifier.eval()

choice = ["A","B","C","D","E"]
test_data = pd.read_csv(test_data_path)
option_tokens = ["False","True"]
token_ids = [tokenizer(o, add_special_tokens=False)["input_ids"][0] for o in option_tokens]
count = 0
for _,row in test_data.iterrows():
    prompts = []
    true_labels = []
    for c in choice:
        prompt = (
                "###\nQuestion: " + row["prompt"] + "\n###\n" +
                "Answer: " + row[c] +
                "\n###\nIs this answer correct? "
        )
        prompts.append(prompt)
        true_label = c == row["answer"]
        true_labels.append(true_label)
    encoded = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to("cuda")
    output = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    last_token_index = encoded["attention_mask"].sum(dim=1) - 1
    batch_size = encoded["input_ids"].size(0)
    vocab_logits = output.logits
    final_logits = vocab_logits[torch.arange(batch_size), last_token_index,:]
    new_poolings = []
    probs = final_logits
    for i in range(len(probs)):
        other_probs = torch.stack([probs[j] for j in range(len(probs)) if j != i])
        new_poolings.append(
            torch.cat(
                [probs[i], torch.mean(other_probs, dim=0)]
            )
        )
    input_tensor = torch.stack(new_poolings).to("cuda")
    input_tensor = input_tensor.to(torch.float32)
    with torch.no_grad():
        logits = binary_classifier(input_tensor)  # [5]
        probs = torch.sigmoid(logits)
        print(probs)
        print(true_labels)
    count += 1
    if count == 5:
        break

# TODO full dataset test