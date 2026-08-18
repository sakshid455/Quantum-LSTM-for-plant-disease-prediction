
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PREDICTIONS_PATH = "outputs/predictions.csv"
TEST_PATH = "data/test.csv"
OUTPUT_DIR = "outputs/prediction_analysis"


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("QLSTM TEST PREDICTION ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_PATH}"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if len(predictions) != len(test_df):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match "
            f"test samples ({len(test_df)})"
        )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_prediction_columns = ["Actual", "Predicted"]

    for column in required_prediction_columns:
        if column not in predictions.columns:
            raise ValueError(
                f"Missing column '{column}' in {PREDICTIONS_PATH}"
            )

    if "target_placl" not in test_df.columns:
        raise ValueError(
            "Column 'target_placl' not found in data/test.csv"
        )

    # --------------------------------------------------------
    # Combine data
    # --------------------------------------------------------

    df = predictions.copy()

    # Use the actual target from test.csv as the authoritative
    # disease-severity target.
    df["Actual"] = test_df["target_placl"].values

    df["Error"] = df["Predicted"] - df["Actual"]
    df["Absolute_Error"] = np.abs(df["Error"])
    df["Squared_Error"] = df["Error"] ** 2

    # --------------------------------------------------------
    # Severity groups
    # --------------------------------------------------------
    # Based on disease-severity ranges used in the project:
    #
    # Low    : < 0.01
    # Medium : 0.01 - < 0.05
    # High   : >= 0.05
    #
    # These are analysis groups, not clinical categories.

    def severity_group(value):

        if value < 0.01:
            return "Low"

        elif value < 0.05:
            return "Medium"

        else:
            return "High"

    df["Severity"] = df["Actual"].apply(severity_group)

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    mse = df["Squared_Error"].mean()
    rmse = np.sqrt(mse)
    mae = df["Absolute_Error"].mean()

    correlation = df["Actual"].corr(df["Predicted"])

    print("\n" + "=" * 70)
    print("OVERALL TEST PERFORMANCE")
    print("=" * 70)

    print(f"Samples       : {len(df)}")
    print(f"Actual mean   : {df['Actual'].mean():.6f}")
    print(f"Predicted mean: {df['Predicted'].mean():.6f}")
    print(f"Actual max    : {df['Actual'].max():.6f}")
    print(f"Predicted max : {df['Predicted'].max():.6f}")
    print(f"MSE           : {mse:.6f}")
    print(f"RMSE          : {rmse:.6f}")
    print(f"MAE           : {mae:.6f}")
    print(f"Correlation   : {correlation:.6f}")

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    mean_error = df["Error"].mean()

    print("\n" + "=" * 70)
    print("BIAS ANALYSIS")
    print("=" * 70)

    print(f"Mean error: {mean_error:.6f}")

    if mean_error > 0:
        print("Interpretation: Model tends to overpredict.")
    elif mean_error < 0:
        print("Interpretation: Model tends to underpredict.")
    else:
        print("Interpretation: No average prediction bias.")

    # --------------------------------------------------------
    # Severity-stratified performance
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEVERITY-STRATIFIED PERFORMANCE")
    print("=" * 70)

    severity_results = []

    for severity in ["Low", "Medium", "High"]:

        group = df[df["Severity"] == severity]

        if len(group) == 0:
            print(f"\n{severity}: No samples")
            continue

        group_mse = group["Squared_Error"].mean()
        group_rmse = np.sqrt(group_mse)
        group_mae = group["Absolute_Error"].mean()
        group_corr = group["Actual"].corr(group["Predicted"])

        print(f"\n{severity}")
        print(f"  Samples    : {len(group)}")
        print(f"  Actual mean: {group['Actual'].mean():.6f}")
        print(f"  Pred mean  : {group['Predicted'].mean():.6f}")
        print(f"  MSE        : {group_mse:.6f}")
        print(f"  RMSE       : {group_rmse:.6f}")
        print(f"  MAE        : {group_mae:.6f}")
        print(f"  Correlation : {group_corr:.6f}")

        severity_results.append({
            "Severity": severity,
            "Samples": len(group),
            "Actual_Mean": group["Actual"].mean(),
            "Predicted_Mean": group["Predicted"].mean(),
            "MSE": group_mse,
            "RMSE": group_rmse,
            "MAE": group_mae,
            "Correlation": group_corr
        })

    severity_df = pd.DataFrame(severity_results)

    severity_df.to_csv(
        f"{OUTPUT_DIR}/severity_performance.csv",
        index=False
    )

    # --------------------------------------------------------
    # Worst predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WORST 10 PREDICTIONS")
    print("=" * 70)

    worst = (
        df.sort_values(
            "Absolute_Error",
            ascending=False
        )
        .head(10)
    )

    print(
        worst[
            [
                "Actual",
                "Predicted",
                "Error",
                "Absolute_Error",
                "Severity"
            ]
        ].to_string(index=False)
    )

    worst.to_csv(
        f"{OUTPUT_DIR}/worst_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # High-severity predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGH-SEVERITY PREDICTIONS")
    print("=" * 70)

    high = df[df["Severity"] == "High"].copy()

    if len(high) > 0:

        print(
            high[
                [
                    "Actual",
                    "Predicted",
                    "Error",
                    "Absolute_Error"
                ]
            ].to_string(index=False)
        )

        high.to_csv(
            f"{OUTPUT_DIR}/high_severity_predictions.csv",
            index=False
        )

    else:

        print("No high-severity samples found.")

    # --------------------------------------------------------
    # Save complete analysis table
    # --------------------------------------------------------

    df.to_csv(
        f"{OUTPUT_DIR}/all_predictions_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot 1: Actual vs Predicted
    # --------------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        df["Actual"],
        df["Predicted"],
        alpha=0.7
    )

    min_value = min(
        df["Actual"].min(),
        df["Predicted"].min()
    )

    max_value = max(
        df["Actual"].max(),
        df["Predicted"].max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Disease Severity")
    plt.ylabel("Predicted Disease Severity")
    plt.title("QLSTM: Actual vs Predicted Disease Severity")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/actual_vs_predicted.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: Prediction errors
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["Error"],
        bins=15,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Number of Samples")
    plt.title("QLSTM Prediction Error Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/error_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: Actual vs Predicted by sample
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    sample_indices = np.arange(len(df))

    plt.plot(
        sample_indices,
        df["Actual"],
        marker="o",
        label="Actual"
    )

    plt.plot(
        sample_indices,
        df["Predicted"],
        marker="x",
        label="Predicted"
    )

    plt.xlabel("Test Sample")
    plt.ylabel("Disease Severity")
    plt.title("QLSTM Test Predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/prediction_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nSaved files:")

    print(
        f"  {OUTPUT_DIR}/all_predictions_analysis.csv"
    )

    print(
        f"  {OUTPUT_DIR}/severity_performance.csv"
    )

    print(
        f"  {OUTPUT_DIR}/worst_predictions.csv"
    )

    if len(high) > 0:
        print(
            f"  {OUTPUT_DIR}/high_severity_predictions.csv"
        )

    print(
        f"  {OUTPUT_DIR}/actual_vs_predicted.png"
    )

    print(
        f"  {OUTPUT_DIR}/error_distribution.png"
    )

    print(
        f"  {OUTPUT_DIR}/prediction_comparison.png"
    )


if __name__ == "__main__":
    main()

