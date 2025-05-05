
# Conver to prompt
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
base_model_path = "deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)
data = pd.read_parquet("sft-data/preprocess/train.parquet")
def get_prompt(data):
    question = data["question"]
    A = data["A"]
    B = data["B"]
    C = data["C"]
    D = data["D"]
    E = data["E"]
    template = f"""Please read the following question and choose the most appropriate answer from the options below. Only respond with the option letter.

        Question: {question}

        Option A. {A}
        Option B. {B}
        Option C. {C}
        Option D. {D}
        Option E. {E}

        Give your answer in <Final Answer>:(your answer)"""
    return template
data["prompt"] = data.apply(get_prompt, axis=1)

def get_batch_origin_answers(df, batch_size=8):
    results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]

        prompts = [get_prompt(row) for _, row in batch.iterrows()]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        for j in range(len(prompts)):
            output_text = tokenizer.decode(
                outputs[j][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            results.append(output_text)

    return results

# 调用
data["origin"] = get_batch_origin_answers(data, batch_size=64)
sft_data = data[["prompt","origin", "answerKey"]]
sft_data.to_parquet("sft-data/preprocess/train_sft.parquet", index=False)