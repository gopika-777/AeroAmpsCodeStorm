import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import unicodedata
import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import nltk
from nltk.stem import WordNetLemmatizer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_PATH = Path(__file__).parents[2]
DATA_PATH = BASE_PATH / "data"
OUTPUTS_PATH = BASE_PATH / "outputs"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = BASE_PATH / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH = BASE_PATH / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_PATH / "trainset.json"
TEST_FILE = DATA_PATH / "testset.json"
DEV_FILE = BASE_PATH / "data" / "dev_testset.json"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VALIDATION_REPORT_PATH = OUTPUTS_PATH / f"validation_report_{timestamp}.txt"

# NLTK path
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
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
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
                "resp_len": len(resp),
                "resp_tokens": len(resp.split())
            }

            if is_train:
                ann = info["annotation"]
                row["Mistake_Identification"] = ann["Mistake_Identification"]
                row["Providing_Guidance"] = ann["Providing_Guidance"]

            rows.append(row)
    return pd.DataFrame(rows)

df_train = prepare_dataframe_all_tutors(train_data, is_train=True)
df_test = prepare_dataframe_all_tutors(test_data, is_train=False)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# ---------------------------------------------------------
# Features & Labels
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    df_train.drop(columns=["Mistake_Identification","Providing_Guidance"]),
    df_train["Providing_Guidance"],
    test_size=0.2,
    random_state=42,
    stratify=df_train["Providing_Guidance"],
)

# ---------------------------------------------------------
# Sentence Embeddings (stronger) + Extra Meta-features
# ---------------------------------------------------------
embed_model = SentenceTransformer('all-mpnet-base-v2')  # 768-dim
X_train_embed = embed_model.encode(X_train["text"].tolist(), show_progress_bar=True, batch_size=64)
X_val_embed = embed_model.encode(X_val["text"].tolist(), show_progress_bar=True, batch_size=64)
X_test_embed = embed_model.encode(df_test["text"].tolist(), show_progress_bar=True, batch_size=64)

def compute_meta_features(df):
    df = df.copy()
    df["num_questions"] = df["response"].str.count(r"\?")
    df["avg_word_len"] = df["response"].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split())>0 else 0)
    df["num_should"] = df["response"].str.lower().str.count("should")
    return df[["resp_len","resp_tokens","num_questions","avg_word_len","num_should"]].to_numpy()

X_train_meta = compute_meta_features(X_train)
X_val_meta = compute_meta_features(X_val)
X_test_meta = compute_meta_features(df_test)

X_train_combined = np.hstack([X_train_embed, X_train_meta])
X_val_combined = np.hstack([X_val_embed, X_val_meta])
X_test_combined = np.hstack([X_test_embed, X_test_meta])

# Standardize meta features (last 5 columns)
scaler = StandardScaler()
X_train_combined[:, -5:] = scaler.fit_transform(X_train_combined[:, -5:])
X_val_combined[:, -5:] = scaler.transform(X_val_combined[:, -5:])
X_test_combined[:, -5:] = scaler.transform(X_test_combined[:, -5:])

# ---------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------
smote = SMOTE(sampling_strategy="not majority", random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_combined, y_train)

# Keep track of label words
label_classes = sorted(y_train_res.unique())
label_to_num = {label:i for i,label in enumerate(label_classes)}
num_to_label = {i: label for i,label in enumerate(label_classes)}
y_train_res_num = np.array([label_to_num[l] for l in y_train_res])
y_val_num = np.array([label_to_num[l] for l in y_val])

# ---------------------------------------------------------
# CatBoost + Random Forest Ensemble
# ---------------------------------------------------------
cb_models = []
for seed in [42, 52]:
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        task_type='GPU',
        devices='0:1',
        early_stopping_rounds=100,
        verbose=100,
        random_seed=seed,
        class_weights=[1.0, 1.5, 1.0]
    )
    model.fit(X_train_res, y_train_res, eval_set=(X_val_combined, y_val))
    cb_models.append(model)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_res, y_train_res)

# ---------------------------------------------------------
# Transformer-based lightweight classifier (PyTorch)
# ---------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=len(label_classes)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train_combined.shape[1]
transformer_clf = TransformerClassifier(input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer_clf.parameters(), lr=1e-3)

X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_res_num, dtype=torch.long).to(device)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

X_val_tensor = torch.tensor(X_val_combined, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_num, dtype=torch.long).to(device)

for epoch in range(10):
    transformer_clf.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = transformer_clf(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# ---------------------------------------------------------
# Soft ensemble prediction
# ---------------------------------------------------------
def ensemble_predict(X_input):
    cb_probs = np.mean([m.predict_proba(X_input) for m in cb_models], axis=0)
    rf_probs = rf_model.predict_proba(X_input)
    X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        tf_probs = torch.softmax(transformer_clf(X_tensor), dim=1).cpu().numpy()
    final_probs = (cb_probs + rf_probs + tf_probs) / 3
    final_labels_num = np.argmax(final_probs, axis=1)
    final_labels_word = [num_to_label[i] for i in final_labels_num]
    return final_labels_word

# ---------------------------------------------------------
# Self-consistency prediction
# ---------------------------------------------------------
def self_consistency_predict_vectorized(X_input, n_samples=5):
    preds_list = []
    for _ in range(n_samples):
        preds_list.append(ensemble_predict(X_input))
    preds_array = np.array(preds_list)
    most_common = [Counter(preds_array[:, i]).most_common(1)[0][0] for i in range(X_input.shape[0])]
    return np.array(most_common)

# ---------------------------------------------------------
# Validation & Confusion Matrix
# ---------------------------------------------------------
y_val_pred = self_consistency_predict_vectorized(X_val_combined, n_samples=5)
acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred, average="macro")
report_str = f"""
Validation Results:
Accuracy: {acc}
Macro F1: {f1}

Classification Report:
{classification_report(y_val, y_val_pred)}
"""
print(report_str)

with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_str)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix (Validation)")
cm_file = OUTPUTS_PATH / f"confusion_matrix_{timestamp}.png"
plt.tight_layout()
plt.savefig(cm_file)
plt.close()

cm = confusion_matrix(y_val, y_val_pred, labels=label_classes)
pd.DataFrame(cm, index=label_classes, columns=label_classes).to_csv(
    OUTPUTS_PATH / f"confusion_matrix_{timestamp}.csv"
)

# ---------------------------------------------------------
# Test predictions
# ---------------------------------------------------------
df_test["predicted_label"] = self_consistency_predict_vectorized(X_test_combined, n_samples=5)
pred_path = OUTPUTS_PATH / f"test_predictions_{timestamp}.csv"
df_test["actual_label"] = df_test.get("Providing_Guidance", "")
df_test[["conversation_id","tutor","actual_label","predicted_label"]].to_csv(pred_path, index=False)
print(f"Test predictions saved to {pred_path}")

# ---------------------------------------------------------
# Dev Set Predictions
# ---------------------------------------------------------
if DEV_FILE.exists():
    dev_data = load_json(DEV_FILE)
    df_dev = prepare_dataframe_all_tutors(dev_data, is_train=False)
    X_dev_embed = embed_model.encode(df_dev["text"].tolist(), show_progress_bar=True, batch_size=128)
    X_dev_meta = compute_meta_features(df_dev)
    X_dev_combined = np.hstack([X_dev_embed, X_dev_meta])
    X_dev_combined[:, -5:] = scaler.transform(X_dev_combined[:, -5:])
    df_dev["predicted_label"] = self_consistency_predict_vectorized(X_dev_combined, n_samples=5)

    updated_dev_data = []
    for item in dev_data:
        convo_id = item["conversation_id"]
        for tutor, info in item["tutor_responses"].items():
            match = df_dev[(df_dev["conversation_id"]==convo_id) & (df_dev["tutor"]==tutor)]
            if not match.empty:
                pred_label = str(match["predicted_label"].values[0]).strip()
                info["annotation"] = {"Providing_Guidance": pred_label}
        updated_dev_data.append(item)

    DEV_RESULTS_FILE = RESULTS_PATH / f"devset_with_predictions_{timestamp}.json"
    with open(DEV_RESULTS_FILE,"w",encoding="utf-8") as f:
        json.dump(updated_dev_data,f,indent=2,ensure_ascii=False)
    print(f"Dev predictions saved to {DEV_RESULTS_FILE}")
