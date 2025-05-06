import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from last_logit_cls import FinalLogitClassifier, CausalLMWithFinalLogitClassification
import torch.nn.functional as F
# 配置路径与参数
base_model_path = "deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
lora_model_path = "./lora-deepseek-sft-v2"
classifier_path = "./deepseek-only-cls/classifier_only.pt"
batch_size = 10
label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

# 加载 tokenizer 和 base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, trust_remote_code=True, device_map="cuda", torch_dtype=torch.float16
)

# 加载 LoRA 和分类器
if_lora = False
if if_lora:
    model = PeftModel.from_pretrained(base_model, lora_model_path).eval()
else:
    model = base_model
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)


for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

def get_prompt(data):
    question = data["prompt"]
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
# 读取测试数据
test_data = pd.read_csv("train.csv")
test_data["question"] = test_data.apply(get_prompt, axis=1)
# 批量生成 <think>
def batch_generate_think(prompts, max_tokens=512):
    think_inputs = prompts
    inputs = tokenizer(think_inputs, return_tensors="pt", padding="longest", truncation=True).to("cuda")
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True
        )
    thinks = []
    for i in range(len(prompts)):
        gen = tokenizer.decode(outputs.sequences[i][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        thinks.append(gen.strip())
    return thinks

# 批处理推理
correct = 0
total = 0
option_tokens = ["A", "B", "C", "D", "E"]
token_ids = [tokenizer(o, add_special_tokens=False)["input_ids"][0] for o in option_tokens]  # [5]

for i in tqdm(range(0, len(test_data), batch_size)):
    batch = test_data.iloc[i:i+batch_size]
    prompts = batch["question"].tolist()
    answers = batch["answer"].tolist()
    # print(prompts[0])
    thinks = batch_generate_think(prompts)
    full_inputs = [p + "\n<think>" + t + "</think>\n" + "Answer: " for p, t in zip(prompts, thinks)]
    # print(full_inputs[0])
    encoded = tokenizer(full_inputs, return_tensors="pt", padding="longest", truncation=True).to("cuda")

    with torch.no_grad():
        output = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        last_token_index = encoded["attention_mask"].sum(dim=1) - 1
        batch_size = encoded["input_ids"].size(0)
        vocab_logits = output.logits
        final_logits = vocab_logits[torch.arange(batch_size), last_token_index]  # [batch, vocab_size]

        logits_for_options = final_logits[:, token_ids]  # [batch, 5]
        probs = F.softmax(logits_for_options, dim=-1)
        pred_indices = torch.argmax(probs, dim=-1).tolist()  # [batch]

    for pred_idx, gold in zip(pred_indices, answers):
        pred = label_map[pred_idx]  # 用已有的 label_map 转换为"A"-"E"
        print(f"[Predicted] {pred} vs [Ground Truth] {gold}")
        if pred == gold:
            correct += 1
        total += 1

# 输出准确率
print("\nCorrect:", correct)
print("Total:", total)
print("Accuracy:", correct / total if total > 0 else 0)
