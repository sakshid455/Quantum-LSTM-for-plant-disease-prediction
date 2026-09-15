import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.quantum.qlstm_model import QLSTMModel
from src.dataset.multimodal_dataloader import create_multimodal_loaders


# ============================================================
# 1. Reproducibility Seed & Device Setup
# ============================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Hyperparameter Configuration
# ============================================================

EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 8

# Balanced multi-task loss weights:
# Normalized disease MSE is ~0.001-0.002, normalized lesion MSE is ~0.05-0.10.
# A 10x weighting on disease balances the gradient magnitudes across both tasks.
DISEASE_LOSS_WEIGHT = 10.0
LESION_LOSS_WEIGHT = 0.5

MAX_GRAD_NORM = 1.0
PATIENCE = 10
HIDDEN_SIZE = 32
USE_MLP_HEADS = True  # Specialized 2-layer MLP projection heads

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("MULTIMODAL QLSTM TRAINING PIPELINE (CORRECTED & DIAGNOSTIC)")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Learning Rate: {LEARNING_RATE}")
print(f"Loss Weights: Disease = {DISEASE_LOSS_WEIGHT} | Lesion = {LESION_LOSS_WEIGHT}")
print(f"QLSTM Hidden Size: {HIDDEN_SIZE} | Specialized MLP Heads: {USE_MLP_HEADS}")
print(f"Early Stopping Patience: {PATIENCE} | Max Grad Norm: {MAX_GRAD_NORM}")


# ============================================================
# 3. Data Loading & Leakage-Free Splitting
# ============================================================

(
    train_loader,
    val_loader,
    test_loader,
    lesion_mean,
    lesion_std
) = create_multimodal_loaders(
    batch_size=BATCH_SIZE,
    random_state=42,
    modality="multimodal"
)

# Extract test split metadata (leaf IDs and sequence indices) for predictions export
raw_data = np.load("data/sequences/multimodal_temporal_sequences.npz", allow_pickle=True)
X_raw = raw_data["X"]
y_disease_raw = raw_data["y_placl"]
y_lesion_raw = raw_data["y_lesion_area"]
leaf_ids_raw = raw_data["leaf_ids"]

gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss_1.split(X_raw, y_disease_raw, groups=leaf_ids_raw))

gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
temp_rel_val, temp_rel_test = next(gss_2.split(temp_idx, y_disease_raw[temp_idx], groups=leaf_ids_raw[temp_idx]))

test_idx = temp_idx[temp_rel_test]
test_leaf_ids = leaf_ids_raw[test_idx]

# Diagnostic statistics printing
sample_batch_x, sample_batch_yd, sample_batch_yl = next(iter(train_loader))
vit_slice = sample_batch_x[:, :, :768]
meta_slice = sample_batch_x[:, :, 768:]

print("\n--- DATASET & FEATURE DIAGNOSTICS ---")
print(f"Multimodal X shape per sequence: {sample_batch_x.shape[1:]} (4 steps, 791 features)")
print(f"  ViT features (0-767)      : Normalized mean={vit_slice.mean():.4f}, std={vit_slice.std():.4f}, min={vit_slice.min():.4f}, max={vit_slice.max():.4f}")
print(f"  Metadata features (768-790): Normalized mean={meta_slice.mean():.4f}, std={meta_slice.std():.4f}, min={meta_slice.min():.4f}, max={meta_slice.max():.4f}")
print(f"Lesion Target Normalization Parameters (Train-Only):")
print(f"  Mean = {lesion_mean:,.2f} px² | Std = {lesion_std:,.2f} px²")
print(f"Disease Target Statistics (Original [0, 1] Scale):")
print(f"  Train mean = {y_disease_raw[train_idx].mean():.4f}, std = {y_disease_raw[train_idx].std():.4f}")
print(f"Test Set: {len(test_idx)} sequences across {len(np.unique(test_leaf_ids))} leaves: {np.unique(test_leaf_ids).tolist()}")


# ============================================================
# 4. Model Instantiation & Parameter Diagnostics
# ============================================================

model = QLSTMModel(
    input_size=791,
    hidden_size=HIDDEN_SIZE,
    mlp_heads=USE_MLP_HEADS,
    dropout=0.1
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable Model Parameters: {total_params:,}")


# ============================================================
# 5. Optimization & Learning Rate Scheduling
# ============================================================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Cosine annealing scheduler to eliminate late-epoch learning rate oscillations
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)

train_losses, val_losses = [], []
train_disease_losses, train_lesion_losses = [], []
val_disease_losses, val_lesion_losses = [], []

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

checkpoint_path = "checkpoints/best_multimodal_qlstm.pth"


# ============================================================
# 6. Training & Validation Loop
# ============================================================

print("\n" + "=" * 70)
print("BEGINNING TRAINING WITH VALIDATION-GUIDED SELECTION")
print("=" * 70)

for epoch in range(EPOCHS):
    model.train()
    r_train_loss, r_train_d, r_train_l = 0.0, 0.0, 0.0

    for X, y_disease, y_lesion in train_loader:
        X = X.to(DEVICE)
        y_disease = y_disease.to(DEVICE)
        y_lesion = y_lesion.to(DEVICE)

        optimizer.zero_grad()
        disease_out, lesion_out = model(X)

        loss_d = criterion(disease_out, y_disease)
        loss_l = criterion(lesion_out, y_lesion)

        # Balanced multi-task loss
        loss = DISEASE_LOSS_WEIGHT * loss_d + LESION_LOSS_WEIGHT * loss_l

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        r_train_loss += loss.item()
        r_train_d += loss_d.item()
        r_train_l += loss_l.item()

    scheduler.step()

    train_loss = r_train_loss / len(train_loader)
    train_d_loss = r_train_d / len(train_loader)
    train_l_loss = r_train_l / len(train_loader)

    # Validation pass (strictly validation data only, zero test set interaction)
    model.eval()
    r_val_loss, r_val_d, r_val_l = 0.0, 0.0, 0.0

    with torch.no_grad():
        for X, y_disease, y_lesion in val_loader:
            X = X.to(DEVICE)
            y_disease = y_disease.to(DEVICE)
            y_lesion = y_lesion.to(DEVICE)

            disease_out, lesion_out = model(X)

            loss_d = criterion(disease_out, y_disease)
            loss_l = criterion(lesion_out, y_lesion)
            loss = DISEASE_LOSS_WEIGHT * loss_d + LESION_LOSS_WEIGHT * loss_l

            r_val_loss += loss.item()
            r_val_d += loss_d.item()
            r_val_l += loss_l.item()

    val_loss = r_val_loss / len(val_loader)
    val_d_loss = r_val_d / len(val_loader)
    val_l_loss = r_val_l / len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_disease_losses.append(train_d_loss)
    train_lesion_losses.append(train_l_loss)
    val_disease_losses.append(val_d_loss)
    val_lesion_losses.append(val_l_loss)

    improved = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), checkpoint_path)
        improved = True
    else:
        epochs_without_improvement += 1

    status_tag = "BEST SAVED" if improved else f"no imp ({epochs_without_improvement}/{PATIENCE})"
    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train: {train_loss:.6f} (D:{train_d_loss:.6f}, L:{train_l_loss:.4f}) | "
        f"Val: {val_loss:.6f} (D:{val_d_loss:.6f}, L:{val_l_loss:.4f}) | "
        f"LR: {current_lr:.2e} | {status_tag}"
    )

    if epochs_without_improvement >= PATIENCE:
        print(f"\n[Early Stopping] Triggered at epoch {epoch+1}.")
        break


# ============================================================
# 7. Save Loss History
# ============================================================

history_df = pd.DataFrame({
    "Epoch": range(1, len(train_losses) + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses,
    "Train Disease Loss": train_disease_losses,
    "Train Lesion Loss": train_lesion_losses,
    "Validation Disease Loss": val_disease_losses,
    "Validation Lesion Loss": val_lesion_losses
})

history_path = "outputs/multimodal_loss_history.csv"
history_df.to_csv(history_path, index=False)
print(f"\nSaved training history -> {history_path}")
print(f"Best Validation Loss: {best_val_loss:.6f} achieved at epoch {best_epoch}")
print(f"Saved best model checkpoint -> {checkpoint_path}")


# ============================================================
# 8. Post-Selection Test Set Evaluation
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION ON UNTOUCHED HELD-OUT LEAVES")
print("=" * 70)

model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

disease_preds, disease_targs = [], []
lesion_preds, lesion_targs = [], []

with torch.no_grad():
    for X, y_d, y_l in test_loader:
        X = X.to(DEVICE)
        d_out, l_out = model(X)

        disease_preds.extend(d_out.cpu().numpy().flatten())
        disease_targs.extend(y_d.numpy().flatten())
        lesion_preds.extend(l_out.cpu().numpy().flatten())
        lesion_targs.extend(y_l.numpy().flatten())

disease_preds = np.array(disease_preds)
disease_targs = np.array(disease_targs)
lesion_preds = np.array(lesion_preds)
lesion_targs = np.array(lesion_targs)

# Reverse lesion normalization using training statistics strictly
lesion_preds_orig = lesion_preds * lesion_std + lesion_mean
lesion_targs_orig = lesion_targs * lesion_std + lesion_mean

# Diagnostic prediction ranges
print("\n--- PREDICTION RANGE DIAGNOSTICS ---")
print(f"Disease Targets (Actual)   : Min = {disease_targs.min():.4f}, Max = {disease_targs.max():.4f}, Mean = {disease_targs.mean():.4f}")
print(f"Disease Predicted          : Min = {disease_preds.min():.4f}, Max = {disease_preds.max():.4f}, Mean = {disease_preds.mean():.4f}")
print(f"Lesion Targets (Actual px²): Min = {lesion_targs_orig.min():,.1f}, Max = {lesion_targs_orig.max():,.1f}, Mean = {lesion_targs_orig.mean():,.1f}")
print(f"Lesion Predicted (px²)     : Min = {lesion_preds_orig.min():,.1f}, Max = {lesion_preds_orig.max():,.1f}, Mean = {lesion_preds_orig.mean():,.1f}")

# Normalized-scale metrics
norm_d_mse = mean_squared_error(disease_targs, disease_preds)
norm_l_mse = mean_squared_error(lesion_targs, lesion_preds)
print(f"\nNormalized-Scale MSE:")
print(f"  Disease Normalized MSE: {norm_d_mse:.6f}")
print(f"  Lesion Normalized MSE : {norm_l_mse:.6f}")

# Original-scale metrics
d_mse = mean_squared_error(disease_targs, disease_preds)
d_rmse = float(d_mse ** 0.5)
d_mae = float(mean_absolute_error(disease_targs, disease_preds))
d_r2 = float(r2_score(disease_targs, disease_preds))

l_mse = mean_squared_error(lesion_targs_orig, lesion_preds_orig)
l_rmse = float(l_mse ** 0.5)
l_mae = float(mean_absolute_error(lesion_targs_orig, lesion_preds_orig))
l_r2 = float(r2_score(lesion_targs_orig, lesion_preds_orig))

metrics_payload = {
    "Disease Severity": {
        "MSE": float(d_mse),
        "RMSE": float(d_rmse),
        "MAE": float(d_mae),
        "R2": float(d_r2)
    },
    "Lesion Area": {
        "MSE": float(l_mse),
        "RMSE": float(l_rmse),
        "MAE": float(l_mae),
        "R2": float(l_r2)
    }
}

metrics_path = "outputs/multimodal_test_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_payload, f, indent=4)
print(f"\nSaved test metrics -> {metrics_path}")

# Export full test predictions CSV with leaf and sequence IDs
predictions_df = pd.DataFrame({
    "leaf_id": test_leaf_ids,
    "sequence_id": test_idx,
    "true_disease": disease_targs,
    "predicted_disease": disease_preds,
    "true_lesion_area": lesion_targs_orig,
    "predicted_lesion_area": lesion_preds_orig
})

pred_path = "outputs/multimodal_predictions.csv"
predictions_df.to_csv(pred_path, index=False)
predictions_df.to_csv("outputs/test_predictions.csv", index=False)
print(f"Saved prediction records -> {pred_path}")

print("\n" + "=" * 70)
print("FINAL TEST METRICS SUMMARY (ORIGINAL SCALES)")
print("=" * 70)
print(f"Disease Severity: MSE = {d_mse:.6f} | RMSE = {d_rmse:.6f} | MAE = {d_mae:.6f} | R² = {d_r2:.6f}")
print(f"Lesion Area     : MSE = {l_mse:,.1f} | RMSE = {l_rmse:,.2f} | MAE = {l_mae:,.2f} | R² = {l_r2:.6f}")
print("=" * 70)