import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import get_peft_model, LoraConfig,PeftModel
from datasets import Dataset
import pandas as pd
import torch.nn.functional as F
from head import SimpleBinaryClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tabulate import tabulate
import numpy as np
from tqdm import tqdm

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
for param in binary_classifier.parameters():
    param.requires_grad = False
for name, param in binary_classifier.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

choice = ["A","B","C","D","E"]
test_data = pd.read_csv(test_data_path)
option_tokens = ["False","True"]
token_ids = [tokenizer(o, add_special_tokens=False)["input_ids"][0] for o in option_tokens]
results = []
predicted_labels = []
true_labels_all = []
for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
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
        pred_idx = torch.argmax(probs).item()
        pred_choice = choice[pred_idx]
        true_choice = row["answer"]

        predicted_labels.append(pred_choice)
        true_labels_all.append(true_choice)
        results.append({
            "question": row["prompt"],
            "true": true_choice,
            "pred": pred_choice,
            "is_correct": pred_choice == true_choice
        })




def evaluate_classifier(true_labels_all, predicted_labels, label_names):
    # 打印整体准确率
    acc = accuracy_score(true_labels_all, predicted_labels)
    print(f"\n✅ Accuracy: {acc:.3f}")

    # 每类准确率
    for idx, label in enumerate(label_names):
        class_mask = np.array(true_labels_all) == idx
        class_acc = accuracy_score(np.array(true_labels_all)[class_mask],
                                   np.array(predicted_labels)[class_mask])
        print(f"Accuracy for label {label}: {class_acc:.3f}")

    # 打印 classification report
    print("\nClassification Report:")
    print(classification_report(true_labels_all, predicted_labels, target_names=label_names, digits=2))

    # 打印 confusion matrix
    cm = confusion_matrix(true_labels_all, predicted_labels)
    print("Confusion Matrix:")
    print(tabulate(cm, headers=label_names, showindex=label_names, tablefmt="fancy_grid"))

label_names = ["A", "B", "C", "D", "E"]
label_map = {c: i for i, c in enumerate(label_names)}
true_ids = [label_map[t] for t in true_labels_all]
pred_ids = [label_map[p] for p in predicted_labels]

evaluate_classifier(true_ids, pred_ids, label_names)