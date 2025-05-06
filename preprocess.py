
# Conver to prompt
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
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

        Your answer:"""
    return template

def get_batch_origin_answers(df,model,tokenizer, batch_size=8):
    results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size]

        prompts = [get_prompt(row) for _, row in batch.iterrows()]

        inputs = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        for j in range(len(prompts)):
            output_text = tokenizer.decode(
                outputs[j][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            ).strip()
            results.append(output_text)

    return results

def shuffle_options_and_update_answer(row):
    options = ["A", "B", "C", "D", "E"]
    original_answer = row["answerKey"]
    option_values = [row[opt] for opt in options]

    # 打乱顺序
    combined = list(zip(options, option_values))
    random.shuffle(combined)

    # 新的选项列和值
    new_options = {chr(65+i): val for i, (_, val) in enumerate(combined)}

    # 找到原答案在打乱后的位置
    original_answer_text = row[original_answer]
    for new_key, value in new_options.items():
        if value == original_answer_text:
            new_answer_key = new_key
            break

    # 更新行
    for key in new_options:
        row[key] = new_options[key]
    row["answerKey"] = new_answer_key
    return row

if __name__ == "__main__":
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
    data = data.apply(shuffle_options_and_update_answer, axis=1)

    data["prompt"] = data.apply(get_prompt, axis=1)

    data["origin"] = get_batch_origin_answers(data,model,tokenizer, batch_size=64)
    sft_data = data[["prompt","origin", "answerKey"]]
    sft_data.to_parquet("sft-data/preprocess/train_sft.parquet", index=False)