import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
import datetime
import unicodedata

# Resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_PATH = Path(__file__).parents[1]
DATA_PATH = BASE_PATH / "data"
OUTPUTS_PATH = BASE_PATH / "outputs"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = BASE_PATH / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

MODELS_PATH = BASE_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_PATH / "trainset.json"
TEST_FILE = DATA_PATH / "testset.json"
DEV_FILE = DATA_PATH / "dev_testset.json"


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VALIDATION_REPORT_PATH = OUTPUTS_PATH / f"validation_report_{timestamp}.txt"

NLTK_MODELS_PATH = BASE_PATH / "models" / "corpora"
nltk.data.path.append(str(NLTK_MODELS_PATH))

# ---------------------------------------------------------
# Chain-of-Thought Prefixes
# ---------------------------------------------------------
cot_prefixes = [
    "Explain step by step and then give the final answer.\nQuestion: ",
    "Reason carefully and provide the solution step by step.\nQuestion: ",
    "Think thoroughly and give your answer with reasoning.\nQuestion: ",
]

# ---------------------------------------------------------
# Text Cleaning / Lemmatization
# ---------------------------------------------------------
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Normalize unicode (e.g., convert full-width to ASCII, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Replace any unicode spaces (incl. \u00a0, \u200b, etc.) with a normal space
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    # Strip leading/trailing spaces
    return text.strip()

# ---------------------------------------------------------
# Load JSON
# ---------------------------------------------------------
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_json(TRAIN_FILE)
test_data = load_json(TEST_FILE)

# ---------------------------------------------------------
# Prepare DataFrame using ALL tutors + CoT
# ---------------------------------------------------------
def prepare_dataframe_all_tutors(data, is_train=True):
    rows = []
    for item in data:
        convo_id = item["conversation_id"]
        convo_text = item["conversation_history"]

        for tutor, info in item["tutor_responses"].items():
            resp = info["response"]
            text_input = np.random.choice(cot_prefixes) + convo_text + " " + resp
            text_input = clean_text(text_input)

            row = {
                "conversation_id": convo_id,
                "tutor": tutor,
                "conversation_history": convo_text,
                "response": resp,
                "text": text_input,
            }

            if is_train:
                ann = info["annotation"]
                row["Mistake_Identification"] = ann["Mistake_Identification"]
                row["Providing_Guidance"] = ann["Providing_Guidance"]

            rows.append(row)
    return pd.DataFrame(rows)


# Train and Test DataFrames
df_train = prepare_dataframe_all_tutors(train_data, is_train=True)
df_test = prepare_dataframe_all_tutors(test_data, is_train=False)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# # ---------------------------------------------------------
# # Features & Labels
# # ---------------------------------------------------------
# X_train, X_val, y_train, y_val = train_test_split(
#     df_train["text"],
#     df_train["Providing_Guidance"],  # <-- Changed to Track 2
#     test_size=0.2,
#     random_state=42,
#     stratify=df_train["Providing_Guidance"],  # <-- Changed to Track 2
# )

# ---------------------------------------------------------
# Features & Labels
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    df_train["text"],
    df_train["Mistake_Identification"],  # <-- Reverted to Track 1
    test_size=0.2,
    random_state=42,
    stratify=df_train["Mistake_Identification"],  # <-- Reverted to Track 1
)

vectorizer = TfidfVectorizer(
    max_features=15000, ngram_range=(1, 2), sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(df_test["text"])

# ---------------------------------------------------------
# Resampling + Ensemble Model
# ---------------------------------------------------------
# Base models
log_reg = LogisticRegression(max_iter=500, class_weight="balanced")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=20, class_weight="balanced", random_state=42
)

# Ensemble
ensemble = VotingClassifier(
    estimators=[("lr", log_reg), ("rf", rf)],
    voting="soft"  # use probabilities for better balance
)

# Resampling pipeline: SMOTE (oversample) + undersampling
resample_pipeline = Pipeline(
    steps=[
        ("smote", SMOTE(sampling_strategy="not majority", random_state=42)),
        ("under", RandomUnderSampler(random_state=42)), # uncomment for under sampling
        ("clf", ensemble),
    ]
)

# Fit model
resample_pipeline.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(resample_pipeline, MODELS_PATH / f"ensemble_model_{timestamp}.joblib")
joblib.dump(vectorizer, MODELS_PATH / f"tfidf_vectorizer_{timestamp}.joblib")

# ---------------------------------------------------------
# Cross-Validation (Macro F1)
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    resample_pipeline, X_train_vec, y_train, cv=cv, scoring="f1_macro"
)
print("Cross-validated Macro F1:", cv_scores.mean())

# ---------------------------------------------------------
# Validation Evaluation
# ---------------------------------------------------------
y_val_pred = resample_pipeline.predict(X_val_vec)

acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred, average="macro")
report_str = f"""
Validation Results:
Accuracy: {acc}
Macro F1: {f1}
Cross-validated Macro F1: {cv_scores.mean()}

Classification Report:
{classification_report(y_val, y_val_pred)}
"""

print(report_str)

# Save validation report
with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_str)

# Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_estimator(
    resample_pipeline, X_val_vec, y_val, cmap="Blues", xticks_rotation=45
)
plt.title("Confusion Matrix (Validation)")
cm_file = OUTPUTS_PATH / f"confusion_matrix_{timestamp}.png"
plt.tight_layout()
plt.savefig(cm_file)
plt.close()

# Save confusion matrix as CSV
cm = confusion_matrix(y_val, y_val_pred, labels=resample_pipeline.classes_)
pd.DataFrame(cm, index=resample_pipeline.classes_, columns=resample_pipeline.classes_).to_csv(
    OUTPUTS_PATH / f"confusion_matrix_{timestamp}.csv"
)


# ---------------------------------------------------------
# Self-Consistency Prediction Function
# ---------------------------------------------------------
def self_consistency_predict(clf, vectorizer, texts, n_samples=3):
    all_preds = []
    for text in texts:
        preds = [clf.predict(vectorizer.transform([text]))[0] for _ in range(n_samples)]
        most_common = Counter(preds).most_common(1)[0][0]
        all_preds.append(most_common)
    return np.array(all_preds)


# ---------------------------------------------------------
# Test Predictions (with Self-Consistency)
# ---------------------------------------------------------
df_test["predicted_label"] = self_consistency_predict(
    resample_pipeline, vectorizer, df_test["text"], n_samples=3
)

# # If test set has labels, evaluate
# if "Providing_Guidance" in df_test.columns:  # <-- Changed to Track 2
#     y_test = df_test["Providing_Guidance"]
#     y_test_pred = df_test["predicted_label"]

#     test_acc = accuracy_score(y_test, y_test_pred)
#     test_f1 = f1_score(y_test, y_test_pred, average="macro")
#     print(f"\nTest Results:\nAccuracy: {test_acc}\nMacro F1: {test_f1}")

# # Save CSV with predictions
# pred_path = OUTPUTS_PATH / f"test_predictions_{timestamp}.csv"
# df_test["actual_label"] = df_test.get("Providing_Guidance", "")  # <-- Changed to Track 2
# df_test[["conversation_id", "tutor", "actual_label", "predicted_label"]].to_csv(
#     pred_path, index=False
# )

# ---------------------------------------------------------
# If test set has labels, evaluate
# ---------------------------------------------------------
if "Mistake_Identification" in df_test.columns:  # <-- Reverted to Track 1
    y_test = df_test["Mistake_Identification"]
    y_test_pred = df_test["predicted_label"]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    print(f"\nTest Results:\nAccuracy: {test_acc}\nMacro F1: {test_f1}")

# Save CSV with predictions
pred_path = OUTPUTS_PATH / f"test_predictions_{timestamp}.csv"
df_test["actual_label"] = df_test.get("Mistake_Identification", "")  # <-- Reverted to Track 1
df_test[["conversation_id", "tutor", "actual_label", "predicted_label"]].to_csv(
    pred_path, index=False
)


# # ---------------------------------------------------------
# # Dev Set Predictions
# # ---------------------------------------------------------
# if DEV_FILE.exists():
#     dev_data = load_json(DEV_FILE)
#     df_dev = prepare_dataframe_all_tutors(dev_data, is_train=False)

#     # Predict with self-consistency
#     df_dev["predicted_label"] = self_consistency_predict(
#         resample_pipeline, vectorizer, df_dev["text"], n_samples=3
#     )

#     # Attach predictions back to JSON structure
#     updated_dev_data = []
#     for item in dev_data:
#         convo_id = item["conversation_id"]
#         for tutor, info in item["tutor_responses"].items():
#             match = df_dev[
#                 (df_dev["conversation_id"] == convo_id) &
#                 (df_dev["tutor"] == tutor)
#             ]
#             if not match.empty:
#                 pred_label = match["predicted_label"].values[0]
#                 info["annotation"] = {"Providing_Guidance": pred_label}  # <-- Changed to Track 2
#         updated_dev_data.append(item)

#     # Save as new JSON copy
#     DEV_RESULTS_FILE = RESULTS_PATH / f"devset_with_predictions_{timestamp}.json"
#     with open(DEV_RESULTS_FILE, "w", encoding="utf-8") as f:
#         json.dump(updated_dev_data, f, indent=2, ensure_ascii=False)

#     print(f"\nDev predictions saved to {DEV_RESULTS_FILE}")

# ---------------------------------------------------------
# Dev Set Predictions
# ---------------------------------------------------------
if DEV_FILE.exists():
    dev_data = load_json(DEV_FILE)
    df_dev = prepare_dataframe_all_tutors(dev_data, is_train=False)

    # Predict with self-consistency
    df_dev["predicted_label"] = self_consistency_predict(
        resample_pipeline, vectorizer, df_dev["text"], n_samples=3
    )

    # Attach predictions back to JSON structure
    updated_dev_data = []
    for item in dev_data:
        convo_id = item["conversation_id"]
        for tutor, info in item["tutor_responses"].items():
            match = df_dev[
                (df_dev["conversation_id"] == convo_id) &
                (df_dev["tutor"] == tutor)
            ]
            if not match.empty:
                pred_label = match["predicted_label"].values[0]
                info["annotation"] = {"Mistake_Identification": pred_label}  # <-- Reverted to Track 1
        updated_dev_data.append(item)

    # Save as new JSON copy
    DEV_RESULTS_FILE = RESULTS_PATH / f"devset_with_predictions_{timestamp}.json"
    with open(DEV_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_dev_data, f, indent=2, ensure_ascii=False)

    print(f"\nDev predictions saved to {DEV_RESULTS_FILE}")