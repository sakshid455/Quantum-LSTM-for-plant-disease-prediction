"""
Multi-seed benchmark on the leak-free rds t+1 forecasting dataset.

Trains QLSTM / LSTM / GRU for every seed (each seed = a different leaf-level
split and initialisation). Resumable: runs whose metrics JSON already exists
are skipped. Summarise afterwards with src.evaluation.summarize_rds_benchmark.

    python scripts/run_rds_benchmark.py [--workers 2] [--threads 2]
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SEEDS = [42, 123, 456, 789, 2026]
# Fast classical runs first so partial results are available early.
ARCHITECTURES = ["lstm", "gru", "qlstm"]
LOG_DIR = "outputs/logs/rds_benchmark"


def run_name(arch, seed):
    base = f"multimodal_rds_seed{seed}"
    return base if arch == "qlstm" else f"{arch}_{base}"


def run(arch, seed, threads):
    metrics = f"outputs/ablation_{run_name(arch, seed)}_qlstm_metrics.json"
    if os.path.exists(metrics):
        return f"skip  {arch:5s} seed {seed} (done)"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf8")
    log = os.path.join(LOG_DIR, f"{arch}_seed{seed}.log")
    cmd = [sys.executable, "-m", "src.train.train_qlstm_ablation", "--dataset", "rds",
           "--architecture", arch, "--seed", str(seed)]
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return f"{'done ' if code == 0 else 'FAIL '} {arch:5s} seed {seed} (exit {code}) -> {log}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    jobs = [(a, s) for a in ARCHITECTURES for s in SEEDS]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for msg in pool.map(lambda j: run(*j, args.threads), jobs):
            print(msg, flush=True)
