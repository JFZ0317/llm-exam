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

class SimpleBinaryClassifier(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

if __name__ == "__main__":
    X, y = torch.load("../sft-data/preprocess/binary_cls_data.pt")  # 文件路径根据实际位置调整
    X = X.to(torch.float32).to("cuda")  # 确保训练稳定
    y = y.to(torch.float32).to("cuda")
    hidden_size = X.shape[1]
    print(hidden_size)
    binary_classifier = SimpleBinaryClassifier(hidden_size).to("cuda")
    optimizer = torch.optim.Adam(binary_classifier.parameters(), lr=1e-6)

    # ===== Step 2: 构建 DataLoader =====
    batch_size = 516
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ===== Step 3: 设置训练参数 =====
    pos_weight = torch.tensor([4.0]).to("cuda")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    binary_classifier.train()
    epoch = 0
    # ===== Step 4: 开始训练 =====
    while True:
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch_x, batch_y in loop:
            optimizer.zero_grad()
            logits = binary_classifier(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        if avg_loss <= 0.5:
            break
        epoch += 1

    save_path = "./binary-cls/binary_classifier.pt"
    torch.save(binary_classifier.state_dict(), save_path)
    print(f"Save to {save_path}")