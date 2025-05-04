from datasets import load_dataset

dataset = load_dataset("ai2_arc", "ARC-Challenge", split="train")

print(dataset[0])
