import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load reference embeddings and aligned 50D embeddings
reference_embeddings = np.load("/Users/tunaseckin/Desktop/ML4RG/ML4RG-group-17/lncrna_to_embedding.pkl", allow_pickle=True)
actual_embeddings = np.load("/Users/tunaseckin/Desktop/ML4RG/ML4RG-group-17/node_embeddings_50d.pkl", allow_pickle=True)
actual_embeddings = actual_embeddings.get("lncRNA")

# Load target table
final_df = pd.read_csv("/Users/tunaseckin/Desktop/ML4RG/ML4RG-group-17/final.tsv", sep="\t")

# Define RRA target columns
rra_columns = [
    'LN18.RRAscore', 'LN229.RRAscore', 'A549.RRAscore',
    'NCIH460.RRAscore', 'KP4.RRAscore', 'MIAPACA2.RRAscore'
]

# Build X and Y using aligned embeddings and RRA scores
X = []
Ys = {col: [] for col in rra_columns}
gene_ids = []

for i, (gene, _) in enumerate(reference_embeddings.items()):
    emb = actual_embeddings[i]  # Access by order
    row = final_df[final_df["incoming"] == gene]
    if not row.empty:
        row = row.iloc[0]
        if not row[rra_columns].isnull().any():
            X.append(emb)
            for col in rra_columns:
                Ys[col].append(row[col])
            gene_ids.append(gene)

X = np.array(X).astype(np.float32)
Ys = {col: np.array(labels).astype(np.float32) for col, labels in Ys.items()}

print(f"Input shape: {X.shape}")

# --- Convert RRA Scores to Quartile Labels ---
def convert_to_quartile_labels(Ys_dict):
    Ys_quartiles = {}
    for col, values in Ys_dict.items():
        col_labels = np.digitize(values, bins=[0.25, 0.5, 0.75])
        print(f"{col} label distribution: {np.bincount(col_labels)}")
        for q in range(4):
            q_vals = values[col_labels == q]
            if len(q_vals) > 0:
                print(f"  Quartile {q}: min={q_vals.min():.4f}, max={q_vals.max():.4f}, count={len(q_vals)}")
        Ys_quartiles[col] = col_labels.astype(np.int64)
    return Ys_quartiles

Y_quartiles = convert_to_quartile_labels(Ys)

# --- Define Model ---
class MLPQuartileClassifier(nn.Module):
    def __init__(self, input_dim=50):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.model(x)

# --- Training Pipeline ---
def train_and_evaluate(task_name, input_dim=50, device='cuda', patience=15, max_epochs=200):
    print(f"\nTask: {task_name}")

    y_task = Y_quartiles[task_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y_task, test_size=0.2, stratify=y_task, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    full_train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    train_labels = y_train_tensor[train_ds.indices].numpy()
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(4)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=4)

    model = MLPQuartileClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(yb.numpy())

    print(classification_report(all_true, all_preds, zero_division=0))

# --- Run the RRA Classification Tasks ---
device = "cuda" if torch.cuda.is_available() else "cpu"
for name in rra_columns:
    train_and_evaluate(name, input_dim=X.shape[1], device=device)
