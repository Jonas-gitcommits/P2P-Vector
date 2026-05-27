import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import time
import random
import csv
import faiss
from config import NUM_NODES, SUBSET_SIZE, EARLY_STOP_ENABLED, EARLY_STOP_THRESHOLD, EVAL_VARIANT

NUM_QUERIES = 1000
NUM_RUNS = 3
DIMENSION = 128
K = 3
TTL_VALUES = [2, 4, 6, 8, 10]
START_PORT = 5000
ALL_NODES = [f"127.0.0.1:{START_PORT + i}" for i in range(NUM_NODES)]

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


def make_request(query_vector, k, ttl, variant, fanout_k):
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    vec = p2p_pb2.Vector(values=query_bytes)

    best_dist = EARLY_STOP_THRESHOLD if (EARLY_STOP_ENABLED and variant == "global_trim") else 0.0

    return p2p_pb2.SearchRequest(
        query=vec,
        k=k,
        ttl=ttl,
        visited_peers=[],
        sender_ip="127.0.0.1",
        sender_port=9999,
        best_dist_so_far=best_dist,
        fanout_k=fanout_k,
    )


def run_evaluation():
    print("Lade Datensatz...")
    dataset = np.load("dataset.npy").astype(np.float32)
    all_queries = np.load("queries.npy").astype(np.float32)
    full_gt = np.load("ground_truth.npy")
    queries = all_queries[:NUM_QUERIES]
    print(f"  dataset: {dataset.shape}, queries: {queries.shape}")
    print(f"  SUBSET_SIZE={SUBSET_SIZE}, EARLY_STOP_ENABLED={EARLY_STOP_ENABLED}, "
          f"THRESHOLD={EARLY_STOP_THRESHOLD}, EVAL_VARIANT={EVAL_VARIANT}")

    print("Berechne Ground-Truth-Distanzen...")
    gt_max_dists = build_ground_truth_max_dists(dataset, queries, full_gt)
    print(f"  Fertig: {len(gt_max_dists)} Distanzen.\n")

    print("Warte 10s für Gossip-Konvergenz...")
    time.sleep(10)

    alive_nodes = get_alive_nodes(ALL_NODES)
    print(f"Erreichbar: {len(alive_nodes)}/{len(ALL_NODES)}")
    if not alive_nodes:
        print("Keine Knoten erreichbar! Abbruch.")
        return

    channels = {n: grpc.insecure_channel(n) for n in alive_nodes}
    stubs = {n: p2p_pb2_grpc.VectorStoreStub(channels[n]) for n in alive_nodes}

    variants = resolve_variants(EVAL_VARIANT)
    fanout_k = max(K * 4, 20)

    results = {ttl: {v: [] for v in variants} for ttl in TTL_VALUES}

    for run_id in range(NUM_RUNS):
        random.seed(42 + run_id)
        entry_nodes = [random.choice(alive_nodes) for _ in queries]

        for ttl in TTL_VALUES:
            for variant in variants:
                print(f"Run {run_id + 1}/{NUM_RUNS} | TTL={ttl} | variant={variant} "
                      f"({NUM_QUERIES} Queries)...")
                latencies = []
                recalls = []

                for i, query_vector in enumerate(queries):
                    entry_node = entry_nodes[i]
                    gt_max = gt_max_dists[i]
                    request = make_request(query_vector, K, ttl, variant, fanout_k)

                    try:
                        t0 = time.time()
                        response = stubs[entry_node].SearchSimilar(request, timeout=10.0)
                        latencies.append((time.time() - t0) * 1000)
                        matches = sum(1 for d in response.distances[:K] if d <= gt_max + 1e-5)
                        recalls.append(min(matches, K) / K)
                    except grpc.RpcError:
                        pass

                if recalls:
                    results[ttl][variant].append(
                        (np.mean(recalls) * 100, np.mean(latencies))
                    )

    for ch in channels.values():
        ch.close()

    print()
    csv_path = f"results_{EVAL_VARIANT}.csv"
    csv_rows = []

    for ttl in TTL_VALUES:
        for variant in variants:
            run_data = results[ttl][variant]
            if not run_data:
                continue
            recall_vals = [r for r, _ in run_data]
            lat_vals = [l for _, l in run_data]
            avg_recall = np.mean(recall_vals)
            std_recall = np.std(recall_vals)
            avg_lat = np.mean(lat_vals)
            std_lat = np.std(lat_vals)

            print(f"TTL={ttl} [{variant}]: "
                  f"{avg_recall:.2f}±{std_recall:.2f}%, "
                  f"{avg_lat:.1f}±{std_lat:.1f} ms  ")
            csv_rows.append({
                "ttl": ttl,
                "variant": variant,
                "recall_mean": round(avg_recall, 4),
                "recall_std": round(std_recall, 4),
                "latency_mean_ms": round(avg_lat, 4),
                "latency_std_ms": round(std_lat, 4),
            })

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV gespeichert: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
