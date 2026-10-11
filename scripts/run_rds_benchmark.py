"""
Multi-seed experiments on the leak-free rds t+1 forecasting dataset.

  benchmark  5 random leaf-level splits (70/15/15 % of leaves)
  season     train on one season, test on every leaf of the other (both directions)
  fraction   train on 5-50 % of the training leaves; val/test leaves unchanged
             (100 % is the benchmark itself)

Every job trains QLSTM / LSTM / GRU. Resumable: runs whose metrics JSON already
exists are skipped. Classical runs are scheduled first so partial results
arrive early.

    python scripts/run_rds_benchmark.py --experiment season fraction [--workers 3] [--threads 1]
        [--architectures qlstm] [--seeds 42 123] [--fractions 0.1 0.25]
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SEEDS = [42, 123, 456, 789, 2026]
SEASON_SEEDS = [42, 123, 456]
SEASONS = [2023, 2024]
FRACTIONS = [0.05, 0.1, 0.25, 0.5]
ARCHITECTURES = ["lstm", "gru", "qlstm"]
LOG_DIR = "outputs/logs/rds_benchmark"


def run_name(arch, seed, train_season=None, train_fraction=1.0):
    name = "multimodal_rds"
    if train_season is not None:
        name += f"_train{train_season}"
    if train_fraction < 1.0:
        name += f"_frac{train_fraction:g}"
    name += f"_seed{seed}"
    return name if arch == "qlstm" else f"{arch}_{name}"


def jobs_for(experiment, seeds=None, fractions=None):
    if experiment == "benchmark":
        return [dict(seed=s) for s in seeds or SEEDS]
    if experiment == "season":
        return [dict(seed=s, train_season=t) for t in SEASONS for s in seeds or SEASON_SEEDS]
    if experiment == "fraction":
        return [dict(seed=s, train_fraction=f) for f in fractions or FRACTIONS for s in seeds or SEEDS]
    raise ValueError(experiment)


def run(arch, job, threads):
    name = run_name(arch, **job)
    if os.path.exists(f"outputs/ablation_{name}_qlstm_metrics.json"):
        return f"skip  {name} (done)"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf8")
    cmd = [sys.executable, "-m", "src.train.train_qlstm_ablation", "--dataset", "rds",
           "--architecture", arch, "--seed", str(job["seed"])]
    if job.get("train_season") is not None:
        cmd += ["--train_season", str(job["train_season"])]
    if job.get("train_fraction", 1.0) < 1.0:
        cmd += ["--train_fraction", str(job["train_fraction"])]
    log = os.path.join(LOG_DIR, f"{name}.log")
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return f"{'done ' if code == 0 else 'FAIL '} {name} (exit {code}) -> {log}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="+", choices=["benchmark", "season", "fraction"], default=["benchmark"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument("--seeds", nargs="+", type=int, help="subset of seeds (default: all for the experiment)")
    parser.add_argument("--fractions", nargs="+", type=float, help="subset of training fractions")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    jobs = [(a, j) for a in args.architectures for e in args.experiment
            for j in jobs_for(e, args.seeds, args.fractions)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for msg in pool.map(lambda aj: run(*aj, args.threads), jobs):
            print(msg, flush=True)
