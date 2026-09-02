import json


# ============================================================
# Paths
# ============================================================

QLSTM_PATH = "outputs/multimodal_test_metrics.json"
LSTM_PATH = "outputs/multimodal_lstm_test_metrics.json"
GRU_PATH = "outputs/multimodal_gru_test_metrics.json"


# ============================================================
# Load JSON
# ============================================================

def load_json(path):

    with open(path, "r") as f:
        return json.load(f)


qlstm = load_json(QLSTM_PATH)
lstm = load_json(LSTM_PATH)
gru = load_json(GRU_PATH)


# ============================================================
# Helper
# ============================================================

def get_metric(data, task, metric):

    return data[task][metric]


# ============================================================
# Header
# ============================================================

print("=" * 90)
print("QLSTM vs CLASSICAL LSTM vs CLASSICAL GRU")
print("=" * 90)


# ============================================================
# Disease Severity
# ============================================================

print("\nDisease Severity")
print("-" * 90)

print(
    f"{'Metric':<12}"
    f"{'QLSTM':>18}"
    f"{'LSTM':>18}"
    f"{'GRU':>18}"
)

print("-" * 90)

for metric in ["MSE", "RMSE", "MAE", "R2"]:

    q = get_metric(
        qlstm,
        "Disease Severity",
        metric
    )

    l = get_metric(
        lstm,
        "Disease Severity",
        metric
    )

    g = get_metric(
        gru,
        "Disease Severity",
        metric
    )

    print(
        f"{metric:<12}"
        f"{q:>18.6f}"
        f"{l:>18.6f}"
        f"{g:>18.6f}"
    )


# ============================================================
# Lesion Area
# ============================================================

print("\nLesion Area")
print("-" * 90)

print(
    f"{'Metric':<12}"
    f"{'QLSTM':>18}"
    f"{'LSTM':>18}"
    f"{'GRU':>18}"
)

print("-" * 90)

for metric in ["MSE", "RMSE", "MAE", "R2"]:

    q = get_metric(
        qlstm,
        "Lesion Area",
        metric
    )

    l = get_metric(
        lstm,
        "Lesion Area",
        metric
    )

    g = get_metric(
        gru,
        "Lesion Area",
        metric
    )

    print(
        f"{metric:<12}"
        f"{q:>18.6f}"
        f"{l:>18.6f}"
        f"{g:>18.6f}"
    )


# ============================================================
# Interpretation
# ============================================================

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)

print("""
All three models use the same multimodal temporal input,
sequence length, leaf-level train/validation/test split,
and training-only feature normalization.

QLSTM is the proposed quantum-enhanced model.

Classical LSTM and GRU are recurrent baselines.

Disease Severity:
QLSTM achieves substantially better held-out test performance
than both classical baselines, with R² = 0.696379.
The classical LSTM and GRU obtain negative R² values on the
held-out leaves, indicating poor generalization.

Lesion Area:
QLSTM achieves R² = 0.618810, slightly higher than the
classical LSTM (R² = 0.608825) and substantially higher than
the GRU (R² = -1.010597).

Lesion-area performance is evaluated after reversing the
training-set target normalization.

For MSE, RMSE and MAE, lower values are better.
For R², higher values are better.

Overall, the current results support QLSTM as the strongest
model among the three evaluated multimodal temporal models
on this held-out test split.
""")
# ============================================================
# Best model per metric
# ============================================================

print("=" * 90)
print("BEST MODEL BY METRIC")
print("=" * 90)


models = {
    "QLSTM": qlstm,
    "LSTM": lstm,
    "GRU": gru
}


for task in [
    "Disease Severity",
    "Lesion Area"
]:

    print(f"\n{task}")

    for metric in [
        "MSE",
        "RMSE",
        "MAE",
        "R2"
    ]:

        values = {
            name: get_metric(
                data,
                task,
                metric
            )
            for name, data in models.items()
        }

        if metric == "R2":

            best_model = max(
                values,
                key=values.get
            )

        else:

            best_model = min(
                values,
                key=values.get
            )

        print(
            f"{metric:<6}: "
            f"{best_model} "
            f"({values[best_model]:.6f})"
        )