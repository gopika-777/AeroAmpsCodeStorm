# day1_baseline_cpu_train_eval.py
import json
from pathlib import Path
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
BASE_PATH = Path(__file__).parents[1]
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

MODELS_PATH = BASE_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_PATH / "trainset.json"
VALIDATION_REPORT_PATH = RESULTS_PATH / "validation_report.txt"

# -------------------------
# Load dataset
# -------------------------
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

rows = []
for conv in train_data:
    convo_id = conv.get("conversation_id", "")
    history = conv.get("conversation_history", "")
    tutor_responses = conv.get("tutor_responses", {})
    for tutor_name, tutor_data in tutor_responses.items():
        rows.append({
            "conversation_id": convo_id,
            "history": history,
            "ground_truth": tutor_data.get("response", "")
        })

df_train = pd.DataFrame(rows)
print(f"Loaded {len(df_train)} rows from train dataset")

# -------------------------
# Load CPU-friendly model
# -------------------------
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)


# -------------------------
# Inference loop
# -------------------------
results = []
for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
    prompt = row["history"]
    if not prompt:
        continue
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=128,  # only this to avoid warnings
        do_sample=False
    )
    latency = time.time() - start
    out_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    results.append({
        "conversation_id": row["conversation_id"],
        "history": prompt,
        "model_answer": out_text,
        "latency_s": latency,
        "ground_truth": row["ground_truth"]
    })

res_df = pd.DataFrame(results)
res_df.to_csv(RESULTS_PATH / "baseline_results_cpu.csv", index=False)
print("Saved baseline_results_cpu.csv")

# -------------------------
# Quick evaluation: similarity + exact match
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device.type)


def similarity(a, b):
    embs = embed_model.encode([a, b])
    return float(cosine_similarity([embs[0]], [embs[1]])[0, 0])

res_df["sim_score"] = res_df.apply(lambda r: similarity(r["ground_truth"], r["model_answer"]), axis=1)
res_df["exact_match"] = res_df.apply(lambda r: float(r["ground_truth"].strip().lower() == r["model_answer"].strip().lower()), axis=1)

# -------------------------
# Validation metrics & report
# -------------------------
acc = res_df["exact_match"].mean()
report_str = f"""
Validation Results on Train Set:
Accuracy (exact match): {acc}
Average semantic similarity: {res_df['sim_score'].mean()}
"""

print(report_str)
with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_str)
print(f"Saved validation report to {VALIDATION_REPORT_PATH}")

# Confusion matrix (treating exact_match as binary classification)
disp = ConfusionMatrixDisplay.from_predictions(
    y_true=res_df["exact_match"].apply(lambda x: int(x)).values,
    y_pred=res_df["exact_match"].apply(lambda x: int(x)).values,
    cmap="Blues"
)
plt.title("Confusion Matrix (Exact Match on Train Set)")
plt.savefig(RESULTS_PATH / "confusion_matrix_train.png")
plt.close()
print(f"Saved confusion matrix plot to {RESULTS_PATH / 'confusion_matrix_train.png'}")

# Save final CSV with metrics
res_df.to_csv(RESULTS_PATH / "baseline_results_with_metrics_cpu.csv", index=False)
print("Saved baseline_results_with_metrics_cpu.csv")
