import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataset.multimodal_dataloader import create_multimodal_loaders
from src.quantum.qlstm_model import QLSTMModel


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


EXPERIMENT_CONFIGS = {
    1: {
        "name": "Exp 1: Extended Horizon (50 ep) + Cosine Annealing + Baseline Loss Weights",
        "epochs": 50,
        "lr": 1e-4,
        "batch_size": 8,
        "disease_weight": 1.0,
        "lesion_weight": 0.5,
        "max_grad_norm": 1.0,
        "patience": 10,
        "mlp_heads": False,
        "scheduler": "cosine",
        "hidden_size": 32,
    },
    2: {
        "name": "Exp 2: Loss-Balanced Multi-Task (Disease 10x) + Cosine Annealing",
        "epochs": 50,
        "lr": 1e-4,
        "batch_size": 8,
        "disease_weight": 10.0,
        "lesion_weight": 0.5,
        "max_grad_norm": 1.0,
        "patience": 10,
        "mlp_heads": False,
        "scheduler": "cosine",
        "hidden_size": 32,
    },
    3: {
        "name": "Exp 3: Specialized MLP Heads + Loss Balancing + Cosine Annealing",
        "epochs": 50,
        "lr": 1e-4,
        "batch_size": 8,
        "disease_weight": 10.0,
        "lesion_weight": 0.5,
        "max_grad_norm": 1.0,
        "patience": 10,
        "mlp_heads": True,
        "scheduler": "cosine",
        "hidden_size": 32,
    },
}


def run_experiment(exp_id):
    cfg = EXPERIMENT_CONFIGS[exp_id]
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 70)
    print(f"RUNNING EXPERIMENT {exp_id}: {cfg['name']}")
    print("=" * 70)
    print(f"Device: {device} | Epochs: {cfg['epochs']} | LR: {cfg['lr']}")
    print(f"Weights: Disease={cfg['disease_weight']}, Lesion={cfg['lesion_weight']}")
    print(f"MLP Heads: {cfg['mlp_heads']} | Scheduler: {cfg['scheduler']}")

    train_loader, val_loader, test_loader, lesion_mean, lesion_std = create_multimodal_loaders(
        batch_size=cfg["batch_size"], random_state=42
    )

    model = QLSTMModel(
        input_size=791,
        hidden_size=cfg["hidden_size"],
        mlp_heads=cfg["mlp_heads"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-5
        )
    else:
        scheduler = None

    train_losses = []
    val_losses = []
    train_disease_losses = []
    train_lesion_losses = []
    val_disease_losses = []
    val_lesion_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = f"checkpoints/best_qlstm_exp{exp_id}.pth"

    for epoch in range(cfg["epochs"]):
        model.train()
        running_train_loss = 0.0
        running_train_disease = 0.0
        running_train_lesion = 0.0

        for X, y_disease, y_lesion in train_loader:
            X = X.to(device)
            y_disease = y_disease.to(device)
            y_lesion = y_lesion.to(device)

            optimizer.zero_grad()
            disease_out, lesion_out = model(X)

            loss_disease = criterion(disease_out, y_disease)
            loss_lesion = criterion(lesion_out, y_lesion)

            loss = cfg["disease_weight"] * loss_disease + cfg["lesion_weight"] * loss_lesion

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["max_grad_norm"])
            optimizer.step()

            running_train_loss += loss.item()
            running_train_disease += loss_disease.item()
            running_train_lesion += loss_lesion.item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_train_loss / len(train_loader)
        train_d_loss = running_train_disease / len(train_loader)
        train_l_loss = running_train_lesion / len(train_loader)

        # Validation (Strictly no test evaluation during training)
        model.eval()
        running_val_loss = 0.0
        running_val_disease = 0.0
        running_val_lesion = 0.0

        with torch.no_grad():
            for X, y_disease, y_lesion in val_loader:
                X = X.to(device)
                y_disease = y_disease.to(device)
                y_lesion = y_lesion.to(device)

                disease_out, lesion_out = model(X)
                loss_disease = criterion(disease_out, y_disease)
                loss_lesion = criterion(lesion_out, y_lesion)

                loss = cfg["disease_weight"] * loss_disease + cfg["lesion_weight"] * loss_lesion

                running_val_loss += loss.item()
                running_val_disease += loss_disease.item()
                running_val_lesion += loss_lesion.item()

        val_loss = running_val_loss / len(val_loader)
        val_d_loss = running_val_disease / len(val_loader)
        val_l_loss = running_val_lesion / len(val_loader)

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

        status_msg = "BEST SAVED" if improved else f"no imp ({epochs_without_improvement}/{cfg['patience']})"
        print(
            f"Epoch {epoch+1:02d}/{cfg['epochs']} | "
            f"Train: {train_loss:.6f} (D:{train_d_loss:.6f}, L:{train_l_loss:.4f}) | "
            f"Val: {val_loss:.6f} (D:{val_d_loss:.6f}, L:{val_l_loss:.4f}) | "
            f"{status_msg}"
        )

        if epochs_without_improvement >= cfg["patience"]:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch + 1}.")
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
    history_path = f"outputs/qlstm_exp{exp_id}_loss_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nSaved history -> {history_path}")
    print(f"Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoint saved -> {checkpoint_path}")

    # Compute validation metrics for the best saved checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    val_d_preds, val_d_targets, val_l_preds, val_l_targets = [], [], [], []
    with torch.no_grad():
        for X, y_disease, y_lesion in val_loader:
            X = X.to(device)
            d_out, l_out = model(X)
            val_d_preds.extend(d_out.cpu().numpy().flatten())
            val_d_targets.extend(y_disease.numpy().flatten())
            val_l_preds.extend(l_out.cpu().numpy().flatten())
            val_l_targets.extend(y_lesion.numpy().flatten())

    val_d_mse = mean_squared_error(val_d_targets, val_d_preds)
    val_d_mae = mean_absolute_error(val_d_targets, val_d_preds)

    val_l_preds_orig = np.array(val_l_preds) * lesion_std + lesion_mean
    val_l_targets_orig = np.array(val_l_targets) * lesion_std + lesion_mean
    val_l_mse = mean_squared_error(val_l_targets_orig, val_l_preds_orig)
    val_l_mae = mean_absolute_error(val_l_targets_orig, val_l_preds_orig)

    summary = {
        "exp_id": exp_id,
        "name": cfg["name"],
        "parameters": total_params,
        "best_epoch": best_epoch,
        "epochs_completed": len(train_losses),
        "best_val_loss": float(best_val_loss),
        "val_disease_mse": float(val_d_mse),
        "val_disease_mae": float(val_d_mae),
        "val_lesion_mse": float(val_l_mse),
        "val_lesion_mae": float(val_l_mae),
    }

    summary_path = f"outputs/qlstm_exp{exp_id}_val_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Validation summary -> {summary_path}")
    return summary


def evaluate_on_test_set(exp_id):
    cfg = EXPERIMENT_CONFIGS[exp_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = f"checkpoints/best_qlstm_exp{exp_id}.pth"
    print("=" * 70)
    print(f"FINAL UNTOUCHED TEST EVALUATION: Exp {exp_id}")
    print("=" * 70)
    print(f"Loading checkpoint: {checkpoint_path}")

    _, _, test_loader, lesion_mean, lesion_std = create_multimodal_loaders(
        batch_size=8, random_state=42
    )

    model = QLSTMModel(
        input_size=791,
        hidden_size=cfg["hidden_size"],
        mlp_heads=cfg["mlp_heads"],
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    d_preds, d_targets = [], []
    l_preds, l_targets = [], []

    with torch.no_grad():
        for X, y_disease, y_lesion in test_loader:
            X = X.to(device)
            d_out, l_out = model(X)
            d_preds.extend(d_out.cpu().numpy().flatten())
            d_targets.extend(y_disease.numpy().flatten())
            l_preds.extend(l_out.cpu().numpy().flatten())
            l_targets.extend(y_lesion.numpy().flatten())

    d_preds = np.array(d_preds)
    d_targets = np.array(d_targets)
    l_preds_orig = np.array(l_preds) * lesion_std + lesion_mean
    l_targets_orig = np.array(l_targets) * lesion_std + lesion_mean

    d_mse = mean_squared_error(d_targets, d_preds)
    d_rmse = d_mse ** 0.5
    d_mae = mean_absolute_error(d_targets, d_preds)
    d_r2 = r2_score(d_targets, d_preds)

    l_mse = mean_squared_error(l_targets_orig, l_preds_orig)
    l_rmse = l_mse ** 0.5
    l_mae = mean_absolute_error(l_targets_orig, l_preds_orig)
    l_r2 = r2_score(l_targets_orig, l_preds_orig)

    test_metrics = {
        "Disease Severity": {
            "MSE": float(d_mse),
            "RMSE": float(d_rmse),
            "MAE": float(d_mae),
            "R2": float(d_r2),
        },
        "Lesion Area": {
            "MSE": float(l_mse),
            "RMSE": float(l_rmse),
            "MAE": float(l_mae),
            "R2": float(l_r2),
        },
    }

    metrics_path = f"outputs/qlstm_exp{exp_id}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Save test predictions CSV
    pred_df = pd.DataFrame({
        "actual_disease": d_targets,
        "predicted_disease": d_preds,
        "actual_lesion": l_targets_orig,
        "predicted_lesion": l_preds_orig,
    })
    pred_path = f"outputs/qlstm_exp{exp_id}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print("\n--- TEST METRICS ---")
    print(f"Disease Severity: MSE={d_mse:.6f}, RMSE={d_rmse:.6f}, MAE={d_mae:.6f}, R2={d_r2:.6f}")
    print(f"Lesion Area     : MSE={l_mse:.1f}, RMSE={l_rmse:.2f}, MAE={l_mae:.2f}, R2={l_r2:.6f}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved predictions -> {pred_path}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], help="Experiment ID to train")
    parser.add_argument("--eval_test", type=int, choices=[1, 2, 3], help="Experiment ID to evaluate on test set")
    args = parser.parse_args()

    if args.exp:
        run_experiment(args.exp)
    elif args.eval_test:
        evaluate_on_test_set(args.eval_test)
    else:
        parser.print_help()
