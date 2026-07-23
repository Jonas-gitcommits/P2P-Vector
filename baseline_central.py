import csv
import importlib
import os
import random
import time

import faiss
import numpy as np
from scipy.stats import t as _t_dist

from evaluate import build_ground_truth_ids   


def _ci95(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(_t_dist.ppf(0.975, n - 1)) * sd / np.sqrt(n)


def main():
    import config as _cfg
    importlib.reload(_cfg)
    from config import (SUBSET_SIZE, NUM_QUERIES, NUM_RUNS, K, DIMENSION,
                        DATASET, SEED)

    print("Lade Datensatz ...")
    dataset = np.load("dataset.npy").astype(np.float32)
    all_queries = np.load("queries.npy").astype(np.float32)
    print(f"  dataset: {dataset.shape}  queries: {all_queries.shape}  "
          f"SUBSET_SIZE={SUBSET_SIZE}  K={K}")

    base = dataset[:SUBSET_SIZE]

    print("Grundwahrheit ...")
    true_ids_all = build_ground_truth_ids(dataset, all_queries, SUBSET_SIZE,
                                          K, DIMENSION, DATASET)

    print("Baue zentrale Indizes ...")
    t0 = time.perf_counter()
    idx_exact = faiss.IndexFlatL2(DIMENSION)
    idx_exact.add(base)
    t_exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx_hnsw = faiss.IndexHNSWFlat(DIMENSION, 16)
    idx_hnsw.hnsw.efConstruction = 200
    idx_hnsw.add(base)
    idx_hnsw.hnsw.efSearch = 64
    t_hnsw = time.perf_counter() - t0
    print(f"  exact: {t_exact:.1f}s   hnsw: {t_hnsw:.1f}s (M=16, efC=200, efS=64)")

    faiss.omp_set_num_threads(1)  

    rows = []
    for arm, index in (("exact", idx_exact), ("hnsw", idx_hnsw)):
        run_recalls, run_lat_means, all_lats = [], [], []

        for run_id in range(NUM_RUNS):
            rng = random.Random(SEED + run_id)
            idx = rng.sample(range(len(all_queries)), NUM_QUERIES)
            run_queries = all_queries[idx]
            run_true_ids = [true_ids_all[j] for j in idx]

            q_recalls, q_lats = [], []
            for i in range(NUM_QUERIES):
                q = run_queries[i:i + 1]
                t0 = time.perf_counter()
                _, ids = index.search(q, K)
                q_lats.append((time.perf_counter() - t0) * 1000.0)
                returned = set(int(x) for x in ids[0] if x >= 0)
                q_recalls.append(len(returned & run_true_ids[i]) / K)

            run_recalls.append(float(np.mean(q_recalls)) * 100.0)
            run_lat_means.append(float(np.mean(q_lats)))
            all_lats.extend(q_lats)
            print(f"  [{arm}] Run {run_id + 1}/{NUM_RUNS}: "
                  f"Recall {run_recalls[-1]:.2f} %  Latenz {run_lat_means[-1]:.3f} ms")

        r_mean = float(np.mean(run_recalls)); r_ci = _ci95(run_recalls)
        l_mean = float(np.mean(run_lat_means)); l_ci = _ci95(run_lat_means)
        rows.append({
            "dataset": DATASET, "block": "central", "arm": arm,
            "num_nodes": 1, "vectors_per_node": SUBSET_SIZE,
            "n_runs": NUM_RUNS, "k": K,
            "recall_over_all_mean": round(r_mean, 4),
            "recall_over_all_ci95_low": round(r_mean - r_ci, 4),
            "recall_over_all_ci95_high": round(r_mean + r_ci, 4),
            "latency_mean_ms": round(l_mean, 4),
            "latency_ci95_low": round(l_mean - l_ci, 4),
            "latency_ci95_high": round(l_mean + l_ci, 4),
            "latency_p50_ms": round(float(np.percentile(all_lats, 50)), 4),
            "latency_p95_ms": round(float(np.percentile(all_lats, 95)), 4),
            "latency_p99_ms": round(float(np.percentile(all_lats, 99)), 4),
            
            "rpc_count_mean": 0, "unique_nodes_mean": 1,
            "index_build_s": round(t_exact if arm == "exact" else t_hnsw, 2),
        })
        print(f"  => {arm}: Recall {r_mean:.2f} +/- {r_ci:.2f} %   "
              f"Latenz {l_mean:.3f} ms (p95 {rows[-1]['latency_p95_ms']:.3f})\n")

    out = "results_central.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    main()