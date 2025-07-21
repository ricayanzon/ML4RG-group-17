import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load reference dictionary with gene keys
reference_embeddings = np.load("/Users/tunaseckin/Desktop/ML4RG/ML4RG-group-17/lncrna_to_embedding.pkl", allow_pickle=True)
# Load actual aligned embeddings (same order)
actual_embeddings = np.load("/Users/tunaseckin/Desktop/ML4RG/ML4RG-group-17/node_embeddings_50d.pkl", allow_pickle=True)
actual_embeddings = actual_embeddings.get("lncRNA")
# Load target table
final_df = pd.read_csv("/Users/tunaseckin/Desktop/ML4RG/ML4RG-group-17/final.tsv", sep="\t")

# Define target columns
lfc_columns = [
    'LN18.LFC', 'LN229.LFC', 'A549.LFC',
    'NCIH460.LFC', 'KP4.LFC', 'MIAPACA2.LFC'
]

# Build X and Y for LFC tasks
X = []
Ys = {col: [] for col in lfc_columns}
gene_ids = []

for i, (gene, _) in enumerate(reference_embeddings.items()):
    emb = actual_embeddings[i]  # Access by order
    row = final_df[final_df["incoming"] == gene]
    if not row.empty:
        row = row.iloc[0]
        if not row[lfc_columns].isnull().any():
            X.append(emb)
            for col in lfc_columns:
                Ys[col].append(1 if row[col] > 0 else 0)  # Binary classification
            gene_ids.append(gene)

X = np.array(X).astype(np.float32)
Ys = {col: np.array(labels).astype(np.int64) for col, labels in Ys.items()}

print(f"Input shape: {X.shape}")
for col in lfc_columns:
    print(f"{col} label distribution: {np.bincount(Ys[col])}")

class LFCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim=50):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 25),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(25, 2)  # Output logits for binary classification
        )

    def forward(self, x):
        return self.model(x)

from torch.nn.functional import cross_entropy
from copy import deepcopy

def train_single_lfc_task(X, y, task_name, epochs=100, patience=15, batch_size=2, lr=1e-3):
    dataset = LFCDataset(X, y)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = MLPBinaryClassifier(input_dim=X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Compute pos_weight to handle imbalance
    y_train = torch.tensor([y for _, y in train_set])
    pos_weight_val = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[{task_name}] Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{task_name}] Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(best_model)

    # Evaluation
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(dataset, batch_size=batch_size):
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_true.extend(yb.numpy())
            all_pred.extend(preds)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = float("nan")

    print(f"[{task_name}] F1 = {f1:.3f}, Precision = {precision:.3f}, Recall = {recall:.3f}, ROC-AUC = {auc:.3f}")

for i, task in enumerate(lfc_columns):
    print(f"\nTraining for task: {task}")
    train_single_lfc_task(X, Ys[task], task_name=task)
