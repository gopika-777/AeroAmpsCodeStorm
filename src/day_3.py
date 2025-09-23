import json
import pickle
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import confusion_matrix, classification_report

# -------------------
# Paths
# -------------------
BASE_PATH = Path(__file__).parents[1]
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

MODELS_PATH = BASE_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_PATH / "trainset.json"
QUESTIONS_FILE = DATA_PATH / "questions.json"

VALIDATION_REPORT_PATH = RESULTS_PATH / "validation_report.txt"
INDEX_PATH = MODELS_PATH / "rag_index.faiss"
DOCS_PATH = MODELS_PATH / "rag_docs.pkl"
RESULTS_CSV = RESULTS_PATH / "rag_results.csv"
CONFUSION_MATRIX_PNG = RESULTS_PATH / "confusion_matrix.png"

# -------------------
# Load Docs
# -------------------
with open(TRAIN_FILE) as f:
    data = json.load(f)

if isinstance(data, list):
    docs = [d if isinstance(d, str) else d.get("doc", "") for d in data]
else:
    docs = data.get("docs", [])

# -------------------
# Load Questions
# -------------------
if QUESTIONS_FILE.exists():
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
else:
    questions = []

print(f"Loaded {len(docs)} documents and {len(questions)} questions")

if not docs:
    raise ValueError("No documents found in trainset.json")

# -------------------
# Build FAISS Index
# -------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = embed_model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(doc_embs)

d = doc_embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(doc_embs)

faiss.write_index(index, str(INDEX_PATH))
with open(DOCS_PATH, "wb") as f:
    pickle.dump(docs, f)

print(f"✅ Saved FAISS index to {INDEX_PATH}")

# -------------------
# Load Generator Model
# -------------------
MODEL_NAME = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# -------------------
# Retrieval Function
# -------------------
def retrieve(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]

def rag_answer(question, k=3):
    ctxs = retrieve(question, k)
    context = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(ctxs)])
    prompt = f"Use the following context to answer the question. If not in context, say 'I don't know'.\n\n{context}\n\nQuestion: {question}\nAnswer:"
    return gen(prompt, max_length=256, do_sample=False)[0]["generated_text"]

# -------------------
# Evaluation
# -------------------
if questions:
    results = []
    y_true, y_pred_baseline, y_pred_rag = [], [], []

    for item in questions:
        q = item["q"]
        gold = item["gold"]

        baseline = gen(q, max_length=128, do_sample=False)[0]["generated_text"]
        rag = rag_answer(q)

        # similarity scoring (cosine)
        gold_emb = embed_model.encode(gold, convert_to_tensor=True)
        base_emb = embed_model.encode(baseline, convert_to_tensor=True)
        rag_emb = embed_model.encode(rag, convert_to_tensor=True)

        base_score = util.cos_sim(gold_emb, base_emb).item()
        rag_score = util.cos_sim(gold_emb, rag_emb).item()

        # binary correctness by threshold (0.5)
        y_true.append(1)
        y_pred_baseline.append(1 if base_score >= 0.5 else 0)
        y_pred_rag.append(1 if rag_score >= 0.5 else 0)

        results.append({
            "question": q,
            "gold": gold,
            "baseline": baseline,
            "rag": rag,
            "baseline_score": base_score,
            "rag_score": rag_score,
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"📄 Saved results to {RESULTS_CSV}")

    # -------------------
    # Confusion Matrices
    # -------------------
    cm_baseline = confusion_matrix(y_true, y_pred_baseline, labels=[1, 0])
    cm_rag = confusion_matrix(y_true, y_pred_rag, labels=[1, 0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # baseline
    im0 = axes[0].imshow(cm_baseline, cmap="Blues")
    axes[0].set_title("Baseline Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    for i in range(cm_baseline.shape[0]):
        for j in range(cm_baseline.shape[1]):
            axes[0].text(j, i, cm_baseline[i, j], ha="center", va="center", color="black")

    # rag
    im1 = axes[1].imshow(cm_rag, cmap="Greens")
    axes[1].set_title("RAG Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    for i in range(cm_rag.shape[0]):
        for j in range(cm_rag.shape[1]):
            axes[1].text(j, i, cm_rag[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG)
    print(f"📊 Saved confusion matrix plot to {CONFUSION_MATRIX_PNG}")

    # -------------------
    # Validation Report
    # -------------------
    report_baseline = classification_report(y_true, y_pred_baseline, target_names=["Incorrect", "Correct"])
    report_rag = classification_report(y_true, y_pred_rag, target_names=["Incorrect", "Correct"])

    with open(VALIDATION_REPORT_PATH, "w") as f:
        f.write("Baseline Model Report\n")
        f.write(report_baseline + "\n\n")
        f.write("RAG Model Report\n")
        f.write(report_rag + "\n")

    print(f"✅ Saved validation report to {VALIDATION_REPORT_PATH}")
else:
    print("⚠️ No questions.json found — skipping evaluation. Only index built.")
