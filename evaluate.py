import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import time
import random
import csv
import faiss
from config import (
    NUM_NODES, SUBSET_SIZE, EARLY_STOP_ENABLED, EARLY_STOP_THRESHOLD, EVAL_VARIANT,
    TOXIPROXY_ENABLED, PROXY_PORT_START, REAL_PORT_START,
    NUM_QUERIES, NUM_RUNS, K, TTL_VALUES, DIMENSION, GOSSIP_WARMUP_S,
)
_PORT_START = PROXY_PORT_START if TOXIPROXY_ENABLED else REAL_PORT_START
ALL_NODES = [f"127.0.0.1:{_PORT_START + i}" for i in range(NUM_NODES)]

def build_ground_truth_max_dists(dataset, queries, full_gt):
    central_index = None
    gt_max_dists = []

    for qi, q in enumerate(queries):
        gt_indices_in_subset = [idx for idx in full_gt[qi] if idx < SUBSET_SIZE]

        if len(gt_indices_in_subset) >= K:
            top_k_vecs = dataset[gt_indices_in_subset[:K]]
            dists = np.linalg.norm(top_k_vecs - q, axis=1) ** 2
            gt_max_dists.append(float(dists.max()))
        else:
            if central_index is None:
                central_index = faiss.IndexFlatL2(DIMENSION)
                central_index.add(dataset)
            d, _ = central_index.search(np.array([q], dtype=np.float32), K)
            gt_max_dists.append(float(d[0][-1]))

    return gt_max_dists


def get_alive_nodes(nodes):
    alive = []
    for n in nodes:
        try:
            channel = grpc.insecure_channel(n)
            stub = p2p_pb2_grpc.VectorStoreStub(channel)
            stub.Ping(p2p_pb2.PingRequest(), timeout=1.0)
            alive.append(n)
        except grpc.RpcError:
            pass
        finally:
            channel.close()
    return alive


def resolve_variants(eval_variant):
    if eval_variant == "local_trim":
        return ["local_trim"]
    if eval_variant == "global_trim":
        return ["global_trim"]
    return ["local_trim", "global_trim"]


def make_request(query_vector, k, ttl, variant, fanout_k, early_stop_threshold=0.0):
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    vec = p2p_pb2.Vector(values=query_bytes)

    kth_dist = early_stop_threshold if (early_stop_threshold > 0 and variant == "global_trim") else 0.0

    return p2p_pb2.SearchRequest(
        query=vec,
        k=k,
        ttl=ttl,
        visited_peers=[],
        sender_ip="127.0.0.1",
        sender_port=9999,
        kth_dist=kth_dist,
        fanout_k=fanout_k,
        early_stop_threshold=early_stop_threshold,
    )


def run_evaluation(early_stop_threshold=None, gossip_warmup_s=GOSSIP_WARMUP_S,
                   scenario_label="default"):
    if early_stop_threshold is None:
        early_stop_threshold = EARLY_STOP_THRESHOLD if EARLY_STOP_ENABLED else 0.0

    print("Lade Datensatz...")
    dataset = np.load("dataset.npy").astype(np.float32)
    all_queries = np.load("queries.npy").astype(np.float32)
    full_gt = np.load("ground_truth.npy")
    queries = all_queries[:NUM_QUERIES]
    print(f"  dataset: {dataset.shape}, queries: {queries.shape}")
    print(f"  SUBSET_SIZE={SUBSET_SIZE}, early_stop_threshold={early_stop_threshold}, "
          f"EVAL_VARIANT={EVAL_VARIANT}, scenario={scenario_label}")

    print("Berechne Ground-Truth-Distanzen...")
    gt_max_dists = build_ground_truth_max_dists(dataset, queries, full_gt)
    print(f"  Fertig: {len(gt_max_dists)} Distanzen.\n")

    if gossip_warmup_s > 0:
        print(f"Warte {gossip_warmup_s}s für Gossip-Konvergenz...")
        time.sleep(gossip_warmup_s)

    alive_nodes = get_alive_nodes(ALL_NODES)
    print(f"Erreichbar: {len(alive_nodes)}/{len(ALL_NODES)}")
    if not alive_nodes:
        print("Keine Knoten erreichbar! Abbruch.")
        return []

    channels = {n: grpc.insecure_channel(n) for n in alive_nodes}
    stubs = {n: p2p_pb2_grpc.VectorStoreStub(channels[n]) for n in alive_nodes}

    variants = resolve_variants(EVAL_VARIANT)
    fanout_k = max(K * 4, 20)

    results = {ttl: {v: {"recalls": [], "latencies": [], "rpcs": [], "unique": []} for v in variants}
               for ttl in TTL_VALUES}

    for run_id in range(NUM_RUNS):
        random.seed(42 + run_id)
        entry_nodes = [random.choice(alive_nodes) for _ in queries]

        for ttl in TTL_VALUES:
            for variant in variants:
                print(f"Run {run_id + 1}/{NUM_RUNS} | TTL={ttl} | variant={variant} "
                      f"({NUM_QUERIES} Queries)...")

                for i, query_vector in enumerate(queries):
                    entry_node = entry_nodes[i]
                    gt_max = gt_max_dists[i]
                    request = make_request(query_vector, K, ttl, variant, fanout_k,
                                          early_stop_threshold)

                    try:
                        t0 = time.time()
                        response = stubs[entry_node].SearchSimilar(request, timeout=10.0)
                        results[ttl][variant]["latencies"].append((time.time() - t0) * 1000)
                        matches = sum(1 for d in response.distances[:K] if d <= gt_max + 1e-5)
                        results[ttl][variant]["recalls"].append(min(matches, K) / K)
                        results[ttl][variant]["rpcs"].append(response.rpc_count)
                        results[ttl][variant]["unique"].append(len(response.visited_nodes))
                    except grpc.RpcError:
                        pass

    for ch in channels.values():
        ch.close()

    print()
    csv_rows = []

    for ttl in TTL_VALUES:
        for variant in variants:
            data = results[ttl][variant]
            recalls = data["recalls"]
            latencies = data["latencies"]
            rpcs = data["rpcs"]
            unique = data["unique"]
            if not recalls:
                continue
            n = len(recalls)
            avg_recall  = np.mean(recalls) * 100
            std_recall  = np.std(recalls) * 100
            sem_recall  = std_recall / np.sqrt(n)
            avg_lat     = np.mean(latencies)
            std_lat     = np.std(latencies)
            sem_lat     = std_lat / np.sqrt(n)
            p95_lat     = np.percentile(latencies, 95)
            avg_rpcs    = np.mean(rpcs)
            p95_rpcs    = np.percentile(rpcs, 95)
            avg_unique  = np.mean(unique)
            p95_unique  = np.percentile(unique, 95)

            print(f"TTL={ttl} [{variant}]: "
                  f"{avg_recall:.2f}±{sem_recall:.2f}%  (std={std_recall:.2f}), "
                  f"{avg_lat:.1f}±{sem_lat:.1f} ms  p95={p95_lat:.1f} ms  "
                  f"rpcs={avg_rpcs:.0f} p95={p95_rpcs:.0f}  "
                  f"unique={avg_unique:.1f} p95={p95_unique:.0f}  (n={n})")
            csv_rows.append({
                "scenario": scenario_label,
                "ttl": ttl,
                "variant": variant,
                "n_queries": n,
                "recall_mean":       round(avg_recall, 4),
                "recall_std":        round(std_recall, 4),
                "recall_sem":        round(sem_recall, 4),
                "latency_mean_ms":   round(avg_lat, 4),
                "latency_std_ms":    round(std_lat, 4),
                "latency_sem_ms":    round(sem_lat, 4),
                "latency_p95_ms":    round(p95_lat, 4),
                "rpc_count_mean":    round(avg_rpcs, 2),
                "rpc_count_p95":     round(p95_rpcs, 1),
                "unique_nodes_mean": round(avg_unique, 2),
                "unique_nodes_p95":  round(p95_unique, 1),
            })

    return csv_rows


if __name__ == "__main__":
    rows = run_evaluation()
    if rows:
        csv_path = f"results_{EVAL_VARIANT}.csv"
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV gespeichert: {csv_path}")
