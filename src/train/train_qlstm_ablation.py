import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.quantum.qlstm_model import QLSTMModel


DATA_PATH = "data/sequences/multimodal_temporal_sequences.npz"


class TemporalSequenceDataset(Dataset):
    def __init__(self, X, y_disease, y_lesion):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_disease = torch.tensor(y_disease, dtype=torch.float32).unsqueeze(1)
        self.y_lesion = torch.tensor(y_lesion, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_disease[idx], self.y_lesion[idx]


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_ablation_loaders(modality="multimodal", batch_size=8, random_state=42):
    data = np.load(DATA_PATH, allow_pickle=True)
    X_full = data["X"]
    y_disease = data["y_placl"]
    y_lesion = data["y_lesion_area"]
    leaf_ids = data["leaf_ids"]

    # Select features based on modality
    if modality == "metadata":
        X = X_full[:, :, 768:]  # 23 metadata features
        modality_name = "Metadata-Only (23 features)"
    elif modality == "vit":
        X = X_full[:, :, :768]  # 768 ViT visual features
        modality_name = "ViT-Only (768 features)"
    elif modality == "multimodal":
        X = X_full              # 791 fused features (768 ViT + 23 metadata)
        modality_name = "Multimodal (791 features)"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    print(f"[{modality_name}] Input X shape: {X.shape}")

    # Stage 1 split: 70% train / 30% temp
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss_1.split(X, y_disease, groups=leaf_ids))

    # Stage 2 split: 15% val / 15% test
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    temp_rel_val, temp_rel_test = next(gss_2.split(temp_idx, y_disease[temp_idx], groups=leaf_ids[temp_idx]))

    val_idx = temp_idx[temp_rel_val]
    test_idx = temp_idx[temp_rel_test]

    train_leaves = set(leaf_ids[train_idx])
    val_leaves = set(leaf_ids[val_idx])
    test_leaves = set(leaf_ids[test_idx])

    assert train_leaves.isdisjoint(val_leaves), "Train/Val leaf overlap!"
    assert train_leaves.isdisjoint(test_leaves), "Train/Test leaf overlap!"
    assert val_leaves.isdisjoint(test_leaves), "Val/Test leaf overlap!"

    # Slices
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    disease_train = y_disease[train_idx]
    disease_val = y_disease[val_idx]
    disease_test = y_disease[test_idx]

    lesion_train = y_lesion[train_idx]
    lesion_val = y_lesion[val_idx]
    lesion_test = y_lesion[test_idx]

    # Normalize input features strictly on training data
    n_features = X_train.shape[-1]
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler.fit(X_train_2d)
    X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

    # Normalize lesion targets strictly on training targets
    lesion_mean = float(lesion_train.mean())
    lesion_std = float(lesion_train.std())
    if lesion_std == 0:
        lesion_std = 1.0

    lesion_train_norm = (lesion_train - lesion_mean) / lesion_std
    lesion_val_norm = (lesion_val - lesion_mean) / lesion_std
    lesion_test_norm = (lesion_test - lesion_mean) / lesion_std

    train_dataset = TemporalSequenceDataset(X_train, disease_train, lesion_train_norm)
    val_dataset = TemporalSequenceDataset(X_val, disease_val, lesion_val_norm)
    test_dataset = TemporalSequenceDataset(X_test, disease_test, lesion_test_norm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        lesion_mean,
        lesion_std,
        n_features,
        leaf_ids[test_idx],
    )


CONFIG = {
    "epochs": 50,
    "lr": 1e-4,
    "batch_size": 8,
    "disease_weight": 10.0,
    "lesion_weight": 0.5,
    "max_grad_norm": 1.0,
    "patience": 10,
    "hidden_size": 32,
    "mlp_heads": True,
}


def train_ablation_model(modality):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("\n" + "=" * 70)
    print(f"ABLATION STUDY: TRAINING {modality.upper()} QLSTM")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        test_loader,
        lesion_mean,
        lesion_std,
        input_dim,
        test_leaf_ids,
    ) = create_ablation_loaders(modality=modality, batch_size=CONFIG["batch_size"])

    model = QLSTMModel(
        input_size=input_dim,
        hidden_size=CONFIG["hidden_size"],
        mlp_heads=CONFIG["mlp_heads"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modality: {modality} | Input dimension: {input_dim} | Parameters: {param_count:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-5
    )

    checkpoint_path = f"checkpoints/ablation_{modality}_qlstm.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    train_losses, val_losses = [], []
    train_disease_losses, train_lesion_losses = [], []
    val_disease_losses, val_lesion_losses = [], []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        r_train_loss = 0.0
        r_train_d = 0.0
        r_train_l = 0.0

        for X, y_d, y_l in train_loader:
            X, y_d, y_l = X.to(device), y_d.to(device), y_l.to(device)

            optimizer.zero_grad()
            d_out, l_out = model(X)

            loss_d = criterion(d_out, y_d)
            loss_l = criterion(l_out, y_l)
            loss = CONFIG["disease_weight"] * loss_d + CONFIG["lesion_weight"] * loss_l

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["max_grad_norm"])
            optimizer.step()

            r_train_loss += loss.item()
            r_train_d += loss_d.item()
            r_train_l += loss_l.item()

        scheduler.step()

        train_loss = r_train_loss / len(train_loader)
        train_d_loss = r_train_d / len(train_loader)
        train_l_loss = r_train_l / len(train_loader)

        # Validation (strictly validation data only)
        model.eval()
        r_val_loss = 0.0
        r_val_d = 0.0
        r_val_l = 0.0

        with torch.no_grad():
            for X, y_d, y_l in val_loader:
                X, y_d, y_l = X.to(device), y_d.to(device), y_l.to(device)
                d_out, l_out = model(X)
                loss_d = criterion(d_out, y_d)
                loss_l = criterion(l_out, y_l)
                loss = CONFIG["disease_weight"] * loss_d + CONFIG["lesion_weight"] * loss_l

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

        status = "BEST SAVED" if improved else f"no imp ({epochs_without_improvement}/{CONFIG['patience']})"
        print(
            f"Epoch {epoch+1:02d}/{CONFIG['epochs']} | "
            f"Train: {train_loss:.6f} (D:{train_d_loss:.6f}, L:{train_l_loss:.4f}) | "
            f"Val: {val_loss:.6f} (D:{val_d_loss:.6f}, L:{val_l_loss:.4f}) | "
            f"{status}"
        )

        if epochs_without_improvement >= CONFIG["patience"]:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1}.")
            break

    # Save training history
    history_df = pd.DataFrame({
        "Epoch": range(1, len(train_losses) + 1),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train Disease Loss": train_disease_losses,
        "Train Lesion Loss": train_lesion_losses,
        "Validation Disease Loss": val_disease_losses,
        "Validation Lesion Loss": val_lesion_losses,
    })
    history_df.to_csv(f"outputs/ablation_{modality}_loss_history.csv", index=False)
    print(f"\nSaved history -> outputs/ablation_{modality}_loss_history.csv")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Evaluate the single best validation checkpoint on the untouched test set
    metrics = evaluate_ablation_test(
        modality=modality,
        test_loader=test_loader,
        lesion_mean=lesion_mean,
        lesion_std=lesion_std,
        input_dim=input_dim,
        param_count=param_count,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        epochs_completed=len(train_losses),
        test_leaf_ids=test_leaf_ids,
    )
    return metrics


def evaluate_ablation_test(
    modality,
    test_loader,
    lesion_mean,
    lesion_std,
    input_dim,
    param_count,
    best_epoch,
    best_val_loss,
    epochs_completed,
    test_leaf_ids,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"checkpoints/ablation_{modality}_qlstm.pth"

    model = QLSTMModel(
        input_size=input_dim,
        hidden_size=CONFIG["hidden_size"],
        mlp_heads=CONFIG["mlp_heads"],
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    d_preds, d_targets = [], []
    l_preds, l_targets = [], []

    with torch.no_grad():
        for X, y_d, y_l in test_loader:
            X = X.to(device)
            d_out, l_out = model(X)
            d_preds.extend(d_out.cpu().numpy().flatten())
            d_targets.extend(y_d.numpy().flatten())
            l_preds.extend(l_out.cpu().numpy().flatten())
            l_targets.extend(y_l.numpy().flatten())

    d_preds = np.array(d_preds)
    d_targets = np.array(d_targets)
    l_preds_orig = np.array(l_preds) * lesion_std + lesion_mean
    l_targets_orig = np.array(l_targets) * lesion_std + lesion_mean

    d_mse = float(mean_squared_error(d_targets, d_preds))
    d_rmse = float(d_mse ** 0.5)
    d_mae = float(mean_absolute_error(d_targets, d_preds))
    d_r2 = float(r2_score(d_targets, d_preds))

    l_mse = float(mean_squared_error(l_targets_orig, l_preds_orig))
    l_rmse = float(l_mse ** 0.5)
    l_mae = float(mean_absolute_error(l_targets_orig, l_preds_orig))
    l_r2 = float(r2_score(l_targets_orig, l_preds_orig))

    metrics = {
        "modality": modality,
        "input_features": input_dim,
        "parameters": param_count,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": float(best_val_loss),
        "epochs_completed": epochs_completed,
        "Disease Severity": {
            "MSE": d_mse,
            "RMSE": d_rmse,
            "MAE": d_mae,
            "R2": d_r2,
        },
        "Lesion Area": {
            "MSE": l_mse,
            "RMSE": l_rmse,
            "MAE": l_mae,
            "R2": l_r2,
        },
    }

    metrics_path = f"outputs/ablation_{modality}_qlstm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    preds_df = pd.DataFrame({
        "leaf_id": test_leaf_ids,
        "actual_disease": d_targets,
        "predicted_disease": d_preds,
        "actual_lesion": l_targets_orig,
        "predicted_lesion": l_preds_orig,
    })
    preds_df.to_csv(f"outputs/ablation_{modality}_predictions.csv", index=False)

    print(f"\n--- TEST SET EVALUATION ({modality.upper()}) ---")
    print(f"Disease Severity: R2={d_r2:.6f}, RMSE={d_rmse:.6f}, MAE={d_mae:.6f}, MSE={d_mse:.6f}")
    print(f"Lesion Area     : R2={l_r2:.6f}, RMSE={l_rmse:,.2f}, MAE={l_mae:,.2f}, MSE={l_mse:,.1f}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved predictions -> outputs/ablation_{modality}_predictions.csv")
    return metrics


def copy_or_evaluate_multimodal():
    # If the winning Exp 3 checkpoint exists and matches configuration, preserve it
    exp3_ckpt = "checkpoints/best_qlstm_exp3.pth"
    target_ckpt = "checkpoints/ablation_multimodal_qlstm.pth"

    import shutil
    shutil.copy(exp3_ckpt, target_ckpt)

    # Evaluate on test set
    _, _, test_loader, lesion_mean, lesion_std, input_dim, test_leaf_ids = create_ablation_loaders(
        modality="multimodal", batch_size=8
    )

    with open("outputs/qlstm_exp3_val_summary.json") as f:
        val_summary = json.load(f)

    metrics = evaluate_ablation_test(
        modality="multimodal",
        test_loader=test_loader,
        lesion_mean=lesion_mean,
        lesion_std=lesion_std,
        input_dim=input_dim,
        param_count=15010,
        best_epoch=val_summary["best_epoch"],
        best_val_loss=val_summary["best_val_loss"],
        epochs_completed=val_summary["epochs_completed"],
        test_leaf_ids=test_leaf_ids,
    )
    return metrics


def build_ablation_comparison_table():
    modalities = ["metadata", "vit", "multimodal"]
    rows = []

    for m in modalities:
        m_path = f"outputs/ablation_{m}_qlstm_metrics.json"
        if os.path.exists(m_path):
            with open(m_path) as f:
                d = json.load(f)

            rows.append({
                "Model": f"QLSTM ({m.capitalize()}-Only)" if m != "multimodal" else "QLSTM (Multimodal)",
                "Input": "23 Metadata" if m == "metadata" else ("768 ViT" if m == "vit" else "768 ViT + 23 Metadata (791)"),
                "Disease R2": d["Disease Severity"]["R2"],
                "Disease RMSE": d["Disease Severity"]["RMSE"],
                "Disease MAE": d["Disease Severity"]["MAE"],
                "Disease MSE": d["Disease Severity"]["MSE"],
                "Lesion R2": d["Lesion Area"]["R2"],
                "Lesion RMSE": d["Lesion Area"]["RMSE"],
                "Lesion MAE": d["Lesion Area"]["MAE"],
                "Lesion MSE": d["Lesion Area"]["MSE"],
                "Parameters": d["parameters"],
                "Best_Val_Epoch": d["best_validation_epoch"],
                "Best_Val_Loss": d["best_validation_loss"],
            })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv("outputs/qlstm_ablation_comparison.csv", index=False)
    print("\n" + "=" * 100)
    print("FINAL ABLATION STUDY COMPARISON TABLE")
    print("=" * 100)
    print(comp_df[["Model", "Input", "Disease R2", "Disease RMSE", "Lesion R2", "Lesion RMSE", "Parameters"]].to_string(index=False))
    print(f"\nSaved comparison -> outputs/qlstm_ablation_comparison.csv")
    return comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, choices=["metadata", "vit", "multimodal", "all"], help="Modality to train")
    parser.add_argument("--use_existing_multimodal", action="store_true", help="Copy and evaluate existing winning Exp 3 for multimodal")
    parser.add_argument("--compare_only", action="store_true", help="Build comparison table from existing JSONs")
    args = parser.parse_args()

    if args.compare_only:
        build_ablation_comparison_table()
    elif args.modality == "multimodal":
        if args.use_existing_multimodal:
            copy_or_evaluate_multimodal()
        else:
            train_ablation_model("multimodal")
        build_ablation_comparison_table()
    elif args.modality in ["metadata", "vit"]:
        train_ablation_model(args.modality)
        build_ablation_comparison_table()
    elif args.modality == "all":
        train_ablation_model("metadata")
        train_ablation_model("vit")
        if args.use_existing_multimodal:
            copy_or_evaluate_multimodal()
        else:
            train_ablation_model("multimodal")
        build_ablation_comparison_table()
    else:
        parser.print_help()
