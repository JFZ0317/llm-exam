from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import pandas as pd
import re
base_model_path = "deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
lora_model_path = "lora-deepseek-sft-v2"
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
if_lora = False
if if_lora:
    model = PeftModel.from_pretrained(base_model, lora_model_path)
else:
    model = base_model
model.eval()

no_answer = 0
quest= 0
correct = 0
for row in pd.read_csv("train.csv", chunksize=1):

    line = row.iloc[0].to_dict()
    question = line["prompt"]
    selection = [line[k] for k in ["A", "B", "C", "D", "E"]]
    template = f"""Please read the following question and choose the most appropriate answer from the options below. Only respond with the option letter.

    Question: {question}

    Option A. {selection[0]}
    Option B. {selection[1]}
    Option C. {selection[2]}
    Option D. {selection[3]}
    Option E. {selection[4]}

    Give your answer in <Final Answer>:(your answer)"""
    inputs = tokenizer(template, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    match = re.search(r"<Final Answer>:\s*([A-E])", text)
    print(text)
    if match:
        answer = match.group(1)
        if answer == line["answer"]:
            correct += 1
    else:

        no_answer += 1
    quest += 1

print("correct",correct)
print("no_answer",no_answer)
print("quest",quest)
