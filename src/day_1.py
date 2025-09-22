# day1_baseline_cpu_train.py
import json
from pathlib import Path
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# -------------------------
# Paths
# -------------------------

BASE_PATH = Path(__file__).parents[1]
data_dir = BASE_PATH / "data"
output_dir = BASE_PATH / "results"
output_dir.mkdir(parents=True, exist_ok=True)

MODELS_PATH = BASE_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)  # new folder for models

train_path = data_dir / "trainset.json"
test_path = data_dir / "testset.json"
VALIDATION_REPORT_PATH = output_dir / "validation_report.txt"

# NLTK_MODELS_PATH = BASE_PATH / "models" / "corpora"

# -------------------------
# Load dataset
# -------------------------
with open(train_path, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for conv in data:
    conversation_id = conv.get("conversation_id", "")
    history = conv.get("conversation_history", "")
    tutor_responses = conv.get("tutor_responses", {})
    for tutor_name, tutor_data in tutor_responses.items():
        rows.append({
            "conversation_id": conversation_id,
            "history": history
        })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} rows from train dataset")

# -------------------------
# Load CPU-friendly model
# -------------------------
MODEL_NAME = "google/flan-t5-small"  # faster on CPU
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cpu")  # force CPU
model.to(device)

# -------------------------
# Inference loop
# -------------------------
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["history"]
    if not prompt:
        continue
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=128,  # smaller for CPU speed
        do_sample=False
    )
    
    latency = time.time() - start
    out_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    results.append({
        "conversation_id": row["conversation_id"],
        "history": prompt,
        "model_answer": out_text,
        "latency_s": latency
    })

res_df = pd.DataFrame(results)
res_df.to_csv(output_dir / "baseline_results_cpu.csv", index=False)
print("Saved baseline_results_cpu.csv")

# -------------------------
# Quick evaluation: semantic similarity + exact match
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def similarity(a, b):
    embs = embed_model.encode([a, b])
    return float(cosine_similarity([embs[0]], [embs[1]])[0, 0])

# Use tutor's first response as "ground truth"
def get_ground_truth(conv):
    for tutor_data in conv.get("tutor_responses", {}).values():
        return tutor_data.get("response", "")
    return ""

df["ground_truth"] = [get_ground_truth(conv) for conv in data for _ in conv.get("tutor_responses", {})]

res_df["sim_score"] = res_df.apply(lambda r: similarity(r["ground_truth"], r["model_answer"]), axis=1)
res_df["exact_match"] = res_df.apply(lambda r: float(r["ground_truth"].strip().lower() == r["model_answer"].strip().lower()), axis=1)

print("Accuracy (exact):", res_df["exact_match"].mean())
print("Avg similarity:", res_df["sim_score"].mean())

res_df.to_csv(output_dir / "baseline_results_with_metrics_cpu.csv", index=False)
print("Saved baseline_results_with_metrics_cpu.csv")
