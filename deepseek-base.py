from openai import OpenAI
import pandas as pd
import re

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

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

    Return your answer as Final-Answer: your answer"""

    response = client.completions.create(
        model="deepseek-r1-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        prompt=template,
        temperature=0.0,
        max_tokens=512
    )
    text = response.choices[0].text
    match = re.search(r"Answer:\s*([A-E])", text)
    if match:
        answer = match.group(1)
        if answer == line["answer"]:
            correct += 1
    else:
        print(text)
        no_answer+=1
    quest += 1

print("correct",correct)
print("no_answer",no_answer)
print("quest",quest)

