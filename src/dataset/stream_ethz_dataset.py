"""
Resilient Stream ETH Zurich Longitudinal Wheat Dataset via WebDAV.
Features:
- Automatic Retry & Exponential Backoff on network timeouts.
- Incremental Checkpointing & Auto-Resume: never lose progress if connection drops.
- Direct in-memory streaming into ViT-B/16 (0 bytes of raw PNG saved to disk).
- Fused 791-D multimodal sliding sequences (T=4).
"""

import os
import io
import sys
import time
import argparse
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

TOKEN = "Agn94FpGxtKyLkd"
WEBDAV_BASE = "https://libdrive.ethz.ch/public.php/webdav/processed/ts/"

# ViT Preprocessing transform
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def create_robust_session():
    """Create requests session with connection pooling and retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_remote_leaf_list(session):
    """Query WebDAV to discover all 1,029 remote leaf directories."""
    headers = {"Depth": "1"}
    for attempt in range(4):
        try:
            res = session.request("PROPFIND", WEBDAV_BASE, auth=(TOKEN, ""), headers=headers, timeout=45)
            if res.status_code in [200, 207]:
                root = ET.fromstring(res.content)
                leaf_dirs = []
                for elem in root.iter("{DAV:}response"):
                    href = elem.find("{DAV:}href").text.rstrip("/")
                    name = href.split("/")[-1]
                    if name and name != "ts":
                        leaf_dirs.append(name)
                return sorted(leaf_dirs)
        except Exception as e:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"WebDAV PROPFIND failed: {e}")
    return []


def list_leaf_files(session, leaf_id, subfolder):
    """List filenames in a leaf's remote subfolder with retries."""
    url = f"{WEBDAV_BASE}{leaf_id}/{subfolder}/"
    headers = {"Depth": "1"}
    for attempt in range(3):
        try:
            res = session.request("PROPFIND", url, auth=(TOKEN, ""), headers=headers, timeout=30)
            if res.status_code in [200, 207]:
                root = ET.fromstring(res.content)
                files = []
                for elem in root.iter("{DAV:}response"):
                    href = elem.find("{DAV:}href").text
                    fname = href.split("/")[-1]
                    if fname and not fname.endswith("/"):
                        files.append(fname)
                return sorted(files)
        except Exception:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
    return []


def load_remote_metadata_row(session, leaf_id, timestamp_str):
    """Fetch leaf_data and lesion_data text files with retry handling."""
    leaf_url = f"{WEBDAV_BASE}{leaf_id}/leaf_data/{timestamp_str}_{leaf_id}.txt"
    lesion_url = f"{WEBDAV_BASE}{leaf_id}/lesion_data/{timestamp_str}_{leaf_id}.txt"

    leaf_text = None
    for _ in range(3):
        try:
            res = session.get(leaf_url, auth=(TOKEN, ""), timeout=25)
            if res.status_code == 200:
                leaf_text = res.text
                break
            elif res.status_code == 404:
                return None
        except Exception:
            time.sleep(1)

    if not leaf_text:
        return None

    try:
        leaf_df = pd.read_csv(io.StringIO(leaf_text))
    except Exception:
        return None

    lesion_metrics = {
        "total_lesion_area": 0.0,
        "mean_lesion_area": 0.0,
        "max_lesion_area": 0.0,
        "lesion_count": 0.0,
        "mean_lesion_perimeter": 0.0,
        "mean_lesion_solidity": 1.0,
        "total_lesion_pycn": 0.0,
        "mean_lesion_pycn_density": 0.0,
        "mean_lesion_rust_density": 0.0
    }

    for _ in range(2):
        try:
            res_lesion = session.get(lesion_url, auth=(TOKEN, ""), timeout=20)
            if res_lesion.status_code == 200:
                lesion_df = pd.read_csv(io.StringIO(res_lesion.text))
                if not lesion_df.empty and "area" in lesion_df.columns:
                    lesion_metrics["total_lesion_area"] = float(lesion_df["area"].sum())
                    lesion_metrics["mean_lesion_area"] = float(lesion_df["area"].mean())
                    lesion_metrics["max_lesion_area"] = float(lesion_df["area"].max())
                    lesion_metrics["lesion_count"] = float(len(lesion_df))
                    if "perimeter" in lesion_df.columns:
                        lesion_metrics["mean_lesion_perimeter"] = float(lesion_df["perimeter"].mean())
                    if "solidity" in lesion_df.columns:
                        lesion_metrics["mean_lesion_solidity"] = float(lesion_df["solidity"].mean())
                break
        except Exception:
            time.sleep(1)

    row = leaf_df.iloc[0].to_dict()
    row.update(lesion_metrics)
    return row


def extract_vit_embedding(model, device, img_url, session):
    """
    Download image into memory buffer with up to 4 retries and exponential backoff.
    Runs ViT inference directly in RAM and discards image bytes immediately.
    """
    for attempt in range(4):
        try:
            res = session.get(img_url, auth=(TOKEN, ""), timeout=60)
            if res.status_code == 200:
                image = Image.open(io.BytesIO(res.content)).convert("RGB")
                tensor = vit_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model(tensor)
                return embedding.squeeze().cpu().numpy()
            elif res.status_code == 404:
                return None
        except (requests.exceptions.RequestException, TimeoutError, Exception) as e:
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"\n[Warning] Timeout fetching image {img_url.split('/')[-1]} after 4 retries. Skipping.")
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Resilient Stream ETH Zurich dataset into ViT+Metadata sequences")
    parser.add_argument("--num_leaves", type=int, default=100, help="Number of leaves to process")
    parser.add_argument("--seq_len", type=int, default=4, help="Temporal sequence sliding window length")
    parser.add_argument("--output", type=str, default="data/sequences/multimodal_temporal_sequences_100leaves.npz", help="Output .npz path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading pretrained ViT-B/16 (vit_base_patch16_224)...")
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(device)
    vit_model.eval()

    session = create_robust_session()

    print("Querying ETH Zurich WebDAV server for leaf directories...")
    all_leaves = fetch_remote_leaf_list(session)
    print(f"Found {len(all_leaves)} total unique leaves on remote repository.")

    selected_leaves = all_leaves[:args.num_leaves]
    print(f"Target cohort: {len(selected_leaves)} leaves.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint_file = args.output.replace(".npz", "_checkpoint.npz")

    # Check for existing checkpoint to resume
    all_sequences = []
    all_disease_targets = []
    all_lesion_targets = []
    all_leaf_ids = []
    processed_leaves_set = set()

    if os.path.exists(checkpoint_file):
        try:
            ckpt = np.load(checkpoint_file, allow_pickle=True)
            all_sequences = list(ckpt["X"])
            all_disease_targets = list(ckpt["y_placl"])
            all_lesion_targets = list(ckpt["y_lesion_area"])
            all_leaf_ids = list(ckpt["leaf_ids"])
            processed_leaves_set = set(all_leaf_ids)
            print(f"\n[RESUME] Found existing checkpoint! Resuming with {len(processed_leaves_set)} leaves already saved ({len(all_sequences)} sequences).")
        except Exception as e:
            print(f"[Warning] Could not load checkpoint ({e}). Starting fresh.")

    # Load experimental design metadata if available
    exp_design_path = "data/meta/experimental_design.csv"
    exp_design = pd.read_csv(exp_design_path).set_index("plot_UID") if os.path.exists(exp_design_path) else None

    pbar = tqdm(selected_leaves, desc="Streaming Leaves")
    for leaf_id in pbar:
        if leaf_id in processed_leaves_set:
            continue

        overlay_files = list_leaf_files(session, leaf_id, "overlay")
        if len(overlay_files) <= args.seq_len:
            continue

        plot_uid = leaf_id.rsplit("_", 1)[0]
        fungicide = 0.0
        inoculation = 0.0
        if exp_design is not None and plot_uid in exp_design.index:
            row = exp_design.loc[plot_uid]
            f_val = str(row["fungicide_treatment"]) if not isinstance(row, pd.DataFrame) else str(row["fungicide_treatment"].iloc[0])
            i_val = str(row["inoculation_treatment"]) if not isinstance(row, pd.DataFrame) else str(row["inoculation_treatment"].iloc[0])
            fungicide = 1.0 if ("Fungicide" in f_val and "No" not in f_val) else 0.0
            inoculation = 1.0 if ("Inoculation" in i_val and "No" not in i_val) else 0.0

        leaf_observations = []

        for img_file in overlay_files:
            ts_str = img_file[:15]
            img_url = f"{WEBDAV_BASE}{leaf_id}/overlay/{img_file}"

            vit_feats = extract_vit_embedding(vit_model, device, img_url, session)
            if vit_feats is None:
                continue

            meta_dict = load_remote_metadata_row(session, leaf_id, ts_str)
            if meta_dict is None:
                continue

            # CLEAN METADATA ONLY:
            # la_tot represents total physical leaf blade area (physical scale).
            # All 22 target-derived lesion/damage segmentations are strictly excluded.
            la_tot = float(meta_dict.get("la_tot", 0.0) if not pd.isna(meta_dict.get("la_tot", 0.0)) else 0.0)
            exogenous_meta = np.array([la_tot, fungicide, inoculation], dtype=np.float32)

            placl = float(meta_dict.get("placl", 0.0) if not pd.isna(meta_dict.get("placl", 0.0)) else 0.0)
            lesion_area = float(meta_dict.get("total_lesion_area", 0.0) if not pd.isna(meta_dict.get("total_lesion_area", 0.0)) else 0.0)

            fused_vector = np.concatenate([vit_feats, exogenous_meta])

            leaf_observations.append({
                "features": fused_vector,
                "placl": placl,
                "lesion_area": lesion_area
            })

        # Construct strictly future temporal forecasting windows: Input [t-3, t-2, t-1, t] -> Target [t+1]
        # Only creates a sequence when a genuine future observation exists (len >= seq_len + 1)
        if len(leaf_observations) > args.seq_len:
            for start_idx in range(len(leaf_observations) - args.seq_len):
                input_window = leaf_observations[start_idx : start_idx + args.seq_len]
                future_target = leaf_observations[start_idx + args.seq_len]

                seq_matrix = np.stack([w["features"] for w in input_window])
                target_placl = future_target["placl"]
                target_lesion = future_target["lesion_area"]

                all_sequences.append(seq_matrix)
                all_disease_targets.append(target_placl)
                all_lesion_targets.append(target_lesion)
                all_leaf_ids.append(leaf_id)

            processed_leaves_set.add(leaf_id)

            # Save checkpoint after each completed leaf
            np.savez_compressed(
                checkpoint_file,
                X=np.array(all_sequences, dtype=np.float32),
                y_placl=np.array(all_disease_targets, dtype=np.float32),
                y_lesion_area=np.array(all_lesion_targets, dtype=np.float32),
                leaf_ids=np.array(all_leaf_ids)
            )

        pbar.set_postfix({"Leaves": len(processed_leaves_set), "Seqs": len(all_sequences)})

    if not all_sequences:
        print("Error: No sequences constructed.")
        return

    X = np.array(all_sequences, dtype=np.float32)
    y_placl = np.array(all_disease_targets, dtype=np.float32)
    y_lesion = np.array(all_lesion_targets, dtype=np.float32)
    leaf_ids = np.array(all_leaf_ids)

    print("\n" + "=" * 70)
    print("STREAMING EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Extracted X shape               : {X.shape} (Sequences, Timesteps, Features)")
    print(f"Disease Targets (y_placl) shape  : {y_placl.shape}")
    print(f"Lesion Targets (y_lesion) shape : {y_lesion.shape}")
    print(f"Unique Leaves                   : {len(np.unique(leaf_ids))}")

    # Final destination
    np.savez_compressed(
        args.output,
        X=X,
        y_placl=y_placl,
        y_lesion_area=y_lesion,
        leaf_ids=leaf_ids
    )

    # Clean up interim checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved compressed sequences -> {args.output} ({file_size_mb:.2f} MB)")
    print("Raw PNG images saved to disk: 0 bytes (Pure in-memory streaming!)")


if __name__ == "__main__":
    main()
