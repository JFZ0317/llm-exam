import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)
data = pd.read_parquet('../sft-data/preprocess/train.parquet')
data = pd.melt(
    data,
    id_vars=["question", "answerKey"],
    value_vars=["A", "B", "C","D","E"],
    var_name="option_label",
    value_name="option_text"
)
data["is_correct"] = data["option_label"] == data["answerKey"]
data = data[data["option_text"].str.strip() != ""].sort_values("question")
print(data.head())
data.to_parquet("../sft-data/preprocess/train_binary.parquet", index=False)
