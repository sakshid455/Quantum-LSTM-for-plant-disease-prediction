import json
import os

QLSTM_PATH = "outputs/multimodal_test_metrics.json"
LSTM_PATH = "outputs/multimodal_lstm_test_metrics.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


qlstm = load_json(QLSTM_PATH)
lstm = load_json(LSTM_PATH)


print("=" * 70)
print("QLSTM vs CLASSICAL LSTM")
print("=" * 70)


for task in ["Disease Severity", "Lesion Area"]:

    print(f"\n{task}")
    print("-" * 70)

    print(
        f"{'Metric':<10}"
        f"{'QLSTM':>18}"
        f"{'LSTM':>18}"
    )

    print("-" * 70)

    for metric in ["MSE", "RMSE", "MAE", "R2"]:

        q = qlstm[task][metric]
        l = lstm[task][metric]

        print(
            f"{metric:<10}"
            f"{q:>18.6f}"
            f"{l:>18.6f}"
        )


print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
QLSTM achieved lower test error and substantially better R²
than the classical LSTM on both prediction tasks.

However, both models have negative test R², indicating that
generalization to the held-out leaves remains a major issue.

The next step is therefore to analyze the target distribution
and prediction behavior on the held-out test leaves before
further modifying the architecture.
""")