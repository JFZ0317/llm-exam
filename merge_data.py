import pandas as pd

df_easy = pd.read_parquet("sft-data/easy/train.parquet")
df_challenge = pd.read_parquet("sft-data/challenge/train.parquet")
save_path = "sft-data/preprocess/train.parquet"
def get_choices_column(df):
    choices_text = df["choices"].apply(lambda x: x["text"])
    df["A"] = choices_text.apply(lambda x: x[0] if len(x) > 0 else "")
    df["B"] = choices_text.apply(lambda x: x[1] if len(x) > 1 else "")
    df["C"] = choices_text.apply(lambda x: x[2] if len(x) > 2 else "")
    df["D"] = choices_text.apply(lambda x: x[3] if len(x) > 3 else "")
    df["E"] = choices_text.apply(lambda x: x[4] if len(x) > 4 else "")
    convert_num = {"1":"A","2":"B","3":"C","4":"D"}
    df["answerKey"] = df["answerKey"].apply(lambda x: convert_num[x] if x in convert_num.keys() else x)
    df = df.drop(columns="choices")
    df = df.drop(columns="id")
    return df
df_easy = get_choices_column(df_easy)
df_challenge = get_choices_column(df_challenge)
df_merged = pd.concat([df_easy, df_challenge], ignore_index=True)
df_merged.to_parquet(save_path, index=False)
