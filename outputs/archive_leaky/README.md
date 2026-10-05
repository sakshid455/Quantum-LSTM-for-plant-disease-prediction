# Archived results — target leakage, do not report

Every result in this folder was produced from datasets whose inputs contain the
prediction target. They are kept only as a record of the leakage audit.

## Affected datasets

| Dataset | Problem |
|---|---|
| `data/sequences/multimodal_temporal_sequences.npz` (30 leaves) | Target at the same timestep as the last input; metadata correlates 0.987 (severity) / 0.980 (lesion area) with the target |
| `data/sequences/metadata_temporal_sequences.npz` (30 leaves) | Same 23 metadata columns as above |
| `data/sequences/multimodal_temporal_sequences_100leaves.npz` (100 leaves) | Metadata column 14 at the last step **equals** the lesion-area target in 100% of windows; column 4 correlates 0.9998 with severity |

These files were built by an older version of `src/dataset/stream_ethz_dataset.py`
that used 23 segmentation-derived metadata columns and a contemporaneous target.
The current version of that script is clean.

## Contents

- `metadata_100leaves/` — side1 metadata-only QLSTM / LSTM / GRU runs (R² 0.96–0.99)
- `multimodal_100leaves/` — the R² = 0.934 / 0.867 "headline" run
- `legacy_30leaf_experiments/`, `plots/`, `prediction_analysis/`, `final_model_comparison.csv`, `qlstm_ablation_comparison.csv` — all 30-leaf results (incl. R² = 0.8745 / 0.6752)
- `logs/` — training logs of the side1 runs

## Valid results (left in `outputs/`)

- `rds_*`, `ablation_*_rds_*` — t+1 forecasting from `data_leaf.rds` history (`data/sequences/rds_forecast_sequences.npz`)
- `histfc_*`, `ablation_*_histfc100_*` — the 100-leaf data re-paired to a t+1 target
- `clean_image_*`, `clean_multimodal_*`, `*clean_metadata*` — t+1 forecasting without lesion-derived inputs
