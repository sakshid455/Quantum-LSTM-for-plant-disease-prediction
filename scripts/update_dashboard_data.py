"""
Update app/data.js with the latest clean forecasting results:
1. Clean Image-Only QLSTM (R2 = 0.8690 / 0.7654)
2. Clean Safe-Multimodal QLSTM (R2 = 0.8343 / 0.7455)
3. Leaky Reference baseline clearly labeled as NOT SCIENTIFICALLY VALID
"""

import json
import pandas as pd
import numpy as np

# Load clean predictions
df_img_preds = pd.read_csv("outputs/clean_image_100leaves_predictions.csv")
df_multi_preds = pd.read_csv("outputs/clean_multimodal_100leaves_predictions.csv")

# Load clean loss histories
df_img_history = pd.read_csv("outputs/clean_image_100leaves_loss_history.csv")
df_multi_history = pd.read_csv("outputs/clean_multimodal_100leaves_loss_history.csv")

# Load clean metrics
with open("outputs/clean_image_100leaves_test_metrics.json") as f:
    img_metrics = json.load(f)
with open("outputs/clean_multimodal_100leaves_test_metrics.json") as f:
    multi_metrics = json.load(f)

# Build data structure
data_js = {
    "cleanImageOnly": {
        "title": "Clean Image-Only QLSTM (Valid Forecasting)",
        "input_dim": 768,
        "metrics": {
            "disease_r2": float(img_metrics["Disease Severity"]["R2"]),
            "disease_rmse": float(img_metrics["Disease Severity"]["RMSE"]),
            "disease_mae": float(img_metrics["Disease Severity"]["MAE"]),
            "lesion_r2": float(img_metrics["Lesion Area"]["R2"]),
            "lesion_rmse": float(img_metrics["Lesion Area"]["RMSE"]),
            "lesion_mae": float(img_metrics["Lesion Area"]["MAE"]),
        },
        "sequences_count": len(df_img_preds),
        "leaves_count": int(df_img_preds["leaf_id"].nunique()),
        "predictions": df_img_preds.to_dict(orient="records"),
        "lossHistory": df_img_history.to_dict(orient="records")
    },
    "cleanMultimodal": {
        "title": "Clean Safe-Multimodal QLSTM (Valid Forecasting)",
        "input_dim": 771,
        "metrics": {
            "disease_r2": float(multi_metrics["Disease Severity"]["R2"]),
            "disease_rmse": float(multi_metrics["Disease Severity"]["RMSE"]),
            "disease_mae": float(multi_metrics["Disease Severity"]["MAE"]),
            "lesion_r2": float(multi_metrics["Lesion Area"]["R2"]),
            "lesion_rmse": float(multi_metrics["Lesion Area"]["RMSE"]),
            "lesion_mae": float(multi_metrics["Lesion Area"]["MAE"]),
        },
        "sequences_count": len(df_multi_preds),
        "leaves_count": int(df_multi_preds["leaf_id"].nunique()),
        "predictions": df_multi_preds.to_dict(orient="records"),
        "lossHistory": df_multi_history.to_dict(orient="records")
    },
    "leakyReference": {
        "title": "Leaky Reference Experiment (NOT SCIENTIFICALLY VALID)",
        "warning": "Contemporaneous targets [t-3..t]->t and target-derived lesion segmentation proxies in input",
        "input_dim": 791,
        "metrics": {
            "disease_r2": 0.933909,
            "disease_rmse": 0.034866,
            "disease_mae": 0.022589,
            "lesion_r2": 0.867431,
            "lesion_rmse": 165905.25,
            "lesion_mae": 78798.65,
        },
        "sequences_count": 181,
        "leaves_count": 15
    },
    "benchmarks": [
        {
            "model": "Clean Image-Only QLSTM",
            "type": "Valid Temporal Forecasting (t+1)",
            "input_dim": 768,
            "disease_r2": 0.8690,
            "disease_rmse": 0.0507,
            "disease_mae": 0.0333,
            "lesion_r2": 0.7654,
            "lesion_rmse": 227987,
            "lesion_mae": 112114,
            "status": "VALID"
        },
        {
            "model": "Clean Safe-Multimodal QLSTM",
            "type": "Valid Temporal Forecasting (t+1)",
            "input_dim": 771,
            "disease_r2": 0.8343,
            "disease_rmse": 0.0570,
            "disease_mae": 0.0316,
            "lesion_r2": 0.7455,
            "lesion_rmse": 237466,
            "lesion_mae": 117348,
            "status": "VALID"
        },
        {
            "model": "Leaky Reference Multimodal QLSTM",
            "type": "Contemporaneous (t->t) / Leaky Shortcuts",
            "input_dim": 791,
            "disease_r2": 0.9339,
            "disease_rmse": 0.0349,
            "disease_mae": 0.0226,
            "lesion_r2": 0.8674,
            "lesion_rmse": 165905,
            "lesion_mae": 78799,
            "status": "LEAKY REFERENCE"
        }
    ]
}

# Root level defaults for initial render
data_js["predictions"] = data_js["cleanImageOnly"]["predictions"]
data_js["lossHistory"] = data_js["cleanImageOnly"]["lossHistory"]

js_content = f"// Generated dataset for Wheat Disease Progression Dashboard\nwindow.DASHBOARD_DATA = {json.dumps(data_js, indent=2)};\n"

with open("app/data.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print("Successfully updated app/data.js with clean forecasting results!")
