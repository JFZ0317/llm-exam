import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from last_logit_cls import FinalLogitClassifier, CausalLMWithFinalLogitClassification

# 配置路径与参数
base_model_path = "deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
lora_model_path = "./deepseek-last-cls"
classifier_path = "./deepseek-last-cls/classifier_only.pt"
batch_size = 10
label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

# 加载 tokenizer 和 base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, trust_remote_code=True, device_map="cuda", torch_dtype=torch.float16
).eval()

# 加载 LoRA 和分类器
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
classifier = FinalLogitClassifier(vocab_size=lora_model.config.vocab_size)
classifier.load_state_dict(torch.load(classifier_path))
classifier.to("cuda").half().eval()
for param in classifier.parameters():
    param.requires_grad = False
# 包装成完整模型
model = CausalLMWithFinalLogitClassification(lora_model, classifier).to("cuda").eval()
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
        preds = torch.argmax(softmax(output["logits"], dim=-1), dim=-1).tolist()

    for pred_idx, gold in zip(preds, answers):
        pred = label_map[pred_idx]
        print(f"[Predicted] {pred} vs [Ground Truth] {gold}")
        if pred == gold:
            correct += 1
        total += 1

# 输出准确率
print("\nCorrect:", correct)
print("Total:", total)
print("Accuracy:", correct / total if total > 0 else 0)
