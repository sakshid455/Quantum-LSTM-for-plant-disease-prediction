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
from src.quantum.lstm_model import LSTMModel
from src.quantum.gru_model import GRUModel


# Leak-free t+1 forecasting datasets only. The contemporaneous datasets with
# target-derived inputs live in data/sequences/archive_leaky/ and must not be
# used for reported results (see outputs/archive_leaky/README.md).
DATASETS = {
    # 995 leaves, 20 leaf-level features from data_leaf.rds (primary benchmark)
    "rds": "data/sequences/rds_forecast_sequences.npz",
    # 99 leaves, 768 ViT + 23 lesion/leaf features, re-paired to a t+1 target
    "history100": "data/sequences/history_forecast_sequences_100leaves.npz",
    # 99 leaves, 768 ViT features only
    "clean_image": "data/sequences/clean_image_temporal_sequences_100leaves.npz",
    # 99 leaves, 768 ViT + la_tot + fungicide + inoculation
    "clean_multimodal": "data/sequences/clean_multimodal_temporal_sequences_100leaves.npz",
    # 99 leaves, la_tot + fungicide + inoculation only
    "clean_metadata": "data/sequences/clean_metadata_temporal_sequences_100leaves.npz",
}
DATA_PATH = DATASETS["rds"]
VIT_DIM = 768

ARCHITECTURES = {
    "qlstm": QLSTMModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
}


def build_model(architecture, input_dim, hidden_size, mlp_heads):
    if architecture == "qlstm":
        return QLSTMModel(input_size=input_dim, hidden_size=hidden_size, mlp_heads=mlp_heads)
    # Classical LSTM/GRU baselines share the same (disease_output, lesion_output)
    # interface as QLSTMModel but don't support the mlp_heads option.
    return ARCHITECTURES[architecture](input_size=input_dim, hidden_size=hidden_size)


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


def split_indices(y, leaf_ids, random_state, seasons=None, train_season=None, train_fraction=1.0):
    """
    Leaf-level train/val/test indices.

    Default: 70/15/15 % of leaves (two-stage GroupShuffleSplit).
    train_season: train/val on that season's leaves (85/15), test on every
        window of the other season(s).
    train_fraction: keep a random subset of the training leaves; val/test
        leaves are unchanged, so results stay comparable across fractions.
    """
    if train_season is not None:
        in_season = np.where(seasons == train_season)[0]
        test_idx = np.where(seasons != train_season)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
        rel_train, rel_val = next(gss.split(in_season, y[in_season], groups=leaf_ids[in_season]))
        train_idx, val_idx = in_season[rel_train], in_season[rel_val]
    else:
        # Stage 1 split: 70% train / 30% temp
        gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
        train_idx, temp_idx = next(gss_1.split(y, y, groups=leaf_ids))

        # Stage 2 split: 15% val / 15% test
        gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
        temp_rel_val, temp_rel_test = next(gss_2.split(temp_idx, y[temp_idx], groups=leaf_ids[temp_idx]))
        val_idx = temp_idx[temp_rel_val]
        test_idx = temp_idx[temp_rel_test]

    if train_fraction < 1.0:
        train_leaves = np.unique(leaf_ids[train_idx])
        n_keep = max(1, int(round(train_fraction * len(train_leaves))))
        keep = np.random.default_rng(random_state).choice(train_leaves, n_keep, replace=False)
        train_idx = train_idx[np.isin(leaf_ids[train_idx], keep)]

    return train_idx, val_idx, test_idx


def create_ablation_loaders(modality="multimodal", batch_size=8, random_state=42, data_path=DATA_PATH,
                            train_season=None, train_fraction=1.0):
    data = np.load(data_path, allow_pickle=True)
    X_full = data["X"]
    y_disease = data["y_placl"]
    y_lesion = data["y_lesion_area"]
    leaf_ids = data["leaf_ids"]

    # Select features based on modality. "metadata"/"vit" slice the fused
    # [768 ViT | metadata] layout; every other modality uses all columns.
    if modality in ("metadata", "vit"):
        if X_full.shape[-1] <= VIT_DIM:
            raise ValueError(
                f"--modality {modality} needs a fused ViT+metadata dataset "
                f"(e.g. --dataset history100); {data_path} has {X_full.shape[-1]} features"
            )
    if modality == "metadata":
        X = X_full[:, :, VIT_DIM:]
        modality_name = f"Metadata-Only ({X.shape[-1]} features)"
    elif modality == "vit":
        X = X_full[:, :, :VIT_DIM]
        modality_name = f"ViT-Only ({VIT_DIM} features)"
    elif modality in ("multimodal", "clean_metadata"):
        X = X_full
        modality_name = f"All features ({X.shape[-1]})"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    print(f"[{modality_name}] Input X shape: {X.shape}")

    seasons = data["season"] if train_season is not None else None
    train_idx, val_idx, test_idx = split_indices(
        y_disease, leaf_ids, random_state, seasons=seasons, train_season=train_season, train_fraction=train_fraction
    )
    print(f"Leaves train/val/test: {len(set(leaf_ids[train_idx]))}/{len(set(leaf_ids[val_idx]))}/{len(set(leaf_ids[test_idx]))} "
          f"| windows {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

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


def train_ablation_model(modality, data_path=DATA_PATH, tag=None, architecture="qlstm", seed=None,
                         train_season=None, train_fraction=1.0):
    run_name = modality if tag is None else f"{modality}_{tag}"
    if architecture != "qlstm":
        run_name = f"{architecture}_{run_name}"
    if train_season is not None:
        run_name = f"{run_name}_train{train_season}"
    if train_fraction < 1.0:
        run_name = f"{run_name}_frac{train_fraction:g}"
    # An explicit seed controls both the leaf split and weight initialisation.
    if seed is not None:
        run_name = f"{run_name}_seed{seed}"
    seed = 42 if seed is None else seed

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("\n" + "=" * 70)
    print(f"ABLATION STUDY: TRAINING {run_name.upper()} ({architecture.upper()})")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        test_loader,
        lesion_mean,
        lesion_std,
        input_dim,
        test_leaf_ids,
    ) = create_ablation_loaders(modality=modality, batch_size=CONFIG["batch_size"], random_state=seed, data_path=data_path,
                              train_season=train_season, train_fraction=train_fraction)

    model = build_model(architecture, input_dim, CONFIG["hidden_size"], CONFIG["mlp_heads"]).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modality: {modality} | Architecture: {architecture} | Input dimension: {input_dim} | Parameters: {param_count:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-5
    )

    checkpoint_path = f"checkpoints/ablation_{run_name}_qlstm.pth"
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
    history_df.to_csv(f"outputs/ablation_{run_name}_loss_history.csv", index=False)
    print(f"\nSaved history -> outputs/ablation_{run_name}_loss_history.csv")
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
        run_name=run_name,
        architecture=architecture,
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
    run_name=None,
    architecture="qlstm",
):
    run_name = modality if run_name is None else run_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"checkpoints/ablation_{run_name}_qlstm.pth"

    model = build_model(architecture, input_dim, CONFIG["hidden_size"], CONFIG["mlp_heads"]).to(device)

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
        "architecture": architecture,
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

    metrics_path = f"outputs/ablation_{run_name}_qlstm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    preds_df = pd.DataFrame({
        "leaf_id": test_leaf_ids,
        "actual_disease": d_targets,
        "predicted_disease": d_preds,
        "actual_lesion": l_targets_orig,
        "predicted_lesion": l_preds_orig,
    })
    preds_df.to_csv(f"outputs/ablation_{run_name}_predictions.csv", index=False)

    print(f"\n--- TEST SET EVALUATION ({run_name.upper()}) ---")
    print(f"Disease Severity: R2={d_r2:.6f}, RMSE={d_rmse:.6f}, MAE={d_mae:.6f}, MSE={d_mse:.6f}")
    print(f"Lesion Area     : R2={l_r2:.6f}, RMSE={l_rmse:,.2f}, MAE={l_mae:,.2f}, MSE={l_mse:,.1f}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved predictions -> outputs/ablation_{run_name}_predictions.csv")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=sorted(DATASETS), default="rds", help="Leak-free t+1 forecasting dataset")
    parser.add_argument("--modality", type=str, choices=["multimodal", "metadata", "vit", "clean_metadata", "all"], default="multimodal",
                        help="multimodal = all features of the dataset; metadata/vit slice the fused history100 layout; all = the three modalities on history100")
    parser.add_argument("--architecture", type=str, choices=["qlstm", "lstm", "gru"], default="qlstm", help="Recurrent architecture to train")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the leaf split and initialisation (appended to output names)")
    parser.add_argument("--train_season", type=int, default=None, help="Train on this season's leaves, test on the other season (rds only)")
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of training leaves to keep (val/test unchanged)")
    args = parser.parse_args()

    if args.modality == "clean_metadata":
        args.dataset = "clean_metadata"
    if args.modality == "all":
        for m in ["metadata", "vit", "multimodal"]:
            train_ablation_model(m, data_path=DATASETS["history100"], tag="history100", architecture=args.architecture, seed=args.seed)
    else:
        train_ablation_model(args.modality, data_path=DATASETS[args.dataset], tag=args.dataset, architecture=args.architecture, seed=args.seed,
                             train_season=args.train_season, train_fraction=args.train_fraction)
