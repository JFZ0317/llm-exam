from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3-8B',
    cache_dir='llama-3'  # ← 你想保存到的路径
)
print("模型已下载到:", model_dir)