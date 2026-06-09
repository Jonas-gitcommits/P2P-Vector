import importlib
import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import time
import random
import faiss
import os
import pickle
from scipy.stats import t


def _t_crit(n):
    return float(t.ppf(0.975, n - 1))


def build_ground_truth_ids(dataset, queries, subset_size, k, dimension, dataset_name):
    import hashlib
    h = hashlib.md5(dataset[:subset_size].tobytes()).hexdigest()[:8]
    cache_file = f"gt_cache_{dataset_name}_size{subset_size}_k{k}_{h}.pkl"
    if os.path.exists(cache_file):
        print(f"  [Cache] Lade Ground-Truth aus {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  [Cache Miss] Berechne Ground-Truth via FAISS (wird in {cache_file} gespeichert)...")
    central_index = faiss.IndexFlatL2(dimension)
    central_index.add(dataset[:subset_size])
    dists, indices = central_index.search(queries, k)

    nn1 = dists[:, 0]
    print(f"Ground-Truth-Distanzstatistik (L2²) für {len(queries)} Queries: "
          f"P50={np.percentile(nn1,50):.4f}  P75={np.percentile(nn1,75):.4f}  "
          f"P90={np.percentile(nn1,90):.4f}  P95={np.percentile(nn1,95):.4f}")

    true_ids = [set(indices[i].tolist()) for i in range(len(queries))]
    with open(cache_file, "wb") as f:
        pickle.dump(true_ids, f)
    return true_ids


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


def make_request(query_vector, k, ttl, fanout_k, early_stop_threshold=0.0):
    query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
    vec = p2p_pb2.Vector(values=query_bytes)

    return p2p_pb2.SearchRequest(
        query=vec,
        k=k,
        ttl=ttl,
        visited_peers=[],
        sender_ip="127.0.0.1",
        sender_port=9999,
        kth_dist=0.0,
        fanout_k=fanout_k,
        early_stop_threshold=early_stop_threshold,
    )


def run_evaluation():
    import config as _cfg
    importlib.reload(_cfg)
    from config import (
        NUM_NODES, SUBSET_SIZE, EARLY_STOP_ENABLED, EARLY_STOP_THRESHOLD,
        TOXIPROXY_ENABLED, PROXY_PORT_START, REAL_PORT_START,
        NUM_QUERIES, NUM_RUNS, K, TTL_VALUES, DIMENSION, GOSSIP_WARMUP_S,
        DATASET, LATENCY_SCENARIO, SEED, ROUTING_STRATEGY,
    )

    early_stop_threshold = EARLY_STOP_THRESHOLD if EARLY_STOP_ENABLED else 0.0
    port_start = PROXY_PORT_START if TOXIPROXY_ENABLED else REAL_PORT_START
    all_nodes = [f"127.0.0.1:{port_start + i}" for i in range(NUM_NODES)]

    print("Lade Datensatz...")
    dataset     = np.load("dataset.npy").astype(np.float32)
    all_queries = np.load("queries.npy").astype(np.float32)
    print(f"  dataset: {dataset.shape}, queries verfügbar: {all_queries.shape}")
    print(f"  routing={ROUTING_STRATEGY}  SUBSET_SIZE={SUBSET_SIZE}  "
          f"early_stop={early_stop_threshold}  scenario={LATENCY_SCENARIO}")

    print("Berechne Ground-Truth-IDs...")
    true_ids_all = build_ground_truth_ids(dataset, all_queries, SUBSET_SIZE, K, DIMENSION, DATASET)
    print(f"  Fertig: {len(true_ids_all)} Referenzmengen.\n")

    if GOSSIP_WARMUP_S > 0:
        print(f"Warte {GOSSIP_WARMUP_S}s für Gossip-Konvergenz...")
        time.sleep(GOSSIP_WARMUP_S)

    alive_nodes = get_alive_nodes(all_nodes)
    print(f"Erreichbar: {len(alive_nodes)}/{len(all_nodes)}")
    if not alive_nodes:
        print("Keine Knoten erreichbar! Abbruch.")
        return []

    channels = {n: grpc.insecure_channel(n) for n in alive_nodes}
    stubs    = {n: p2p_pb2_grpc.VectorStoreStub(channels[n]) for n in alive_nodes}

    nb_counts = []
    for stub in stubs.values():
        try:
            nb_counts.append(stub.Ping(p2p_pb2.PingRequest(), timeout=1.0).neighbor_count)
        except grpc.RpcError:
            pass
    if nb_counts:
        print(f"Ø Nachbarn pro Knoten: {np.mean(nb_counts):.1f}  "
              f"(min={min(nb_counts)}, max={max(nb_counts)})")

    fanout_k = max(K * 4, 20)
    run_data = {ttl: {"recalls": [], "latencies": [], "all_lats": [],
                      "rpcs": [], "unique": [], "failures": [], "timeouts": [],
                      "incomplete": [], "recalls_all": []}
                for ttl in TTL_VALUES}

    for run_id in range(NUM_RUNS):
        rng = random.Random(SEED + run_id)
        idx = rng.sample(range(len(all_queries)), NUM_QUERIES)
        run_queries  = all_queries[idx]
        run_true_ids = [true_ids_all[j] for j in idx]
        entry_nodes  = [rng.choice(alive_nodes) for _ in range(NUM_QUERIES)]

        for ttl in TTL_VALUES:
            print(f"Run {run_id + 1}/{NUM_RUNS} | TTL={ttl} ({NUM_QUERIES} Queries)...")
            q_recalls, q_lats, q_rpcs, q_unique = [], [], [], []
            q_failures, q_timeouts, q_incomplete = 0, 0, 0

            for i, query_vector in enumerate(run_queries):
                request = make_request(query_vector, K, ttl, fanout_k, early_stop_threshold)
                try:
                    t0 = time.time()
                    response = stubs[entry_nodes[i]].SearchSimilar(request, timeout=10.0)
                    q_lats.append((time.time() - t0) * 1000)
                    returned_ids = set(response.vector_ids[:K])
                    if len(returned_ids) < K:
                        q_incomplete += 1
                    matches = len(returned_ids & run_true_ids[i])
                    q_recalls.append(matches / K)
                    q_rpcs.append(response.rpc_count)
                    q_unique.append(len(response.visited_nodes))
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        q_timeouts += 1
                    else:
                        q_failures += 1

            assert len(q_recalls) + q_failures + q_timeouts == NUM_QUERIES, (
                f"Run {run_id + 1}, TTL={ttl}: "
                f"success={len(q_recalls)} failures={q_failures} timeouts={q_timeouts} "
                f"!= NUM_QUERIES={NUM_QUERIES}"
            )

            rd = run_data[ttl]
            rd["failures"].append(q_failures)
            rd["timeouts"].append(q_timeouts)
            rd["incomplete"].append(q_incomplete)
            rd["recalls_all"].append(sum(q_recalls) / NUM_QUERIES)
            if q_recalls:
                rd["recalls"].append(np.mean(q_recalls))
                rd["latencies"].append(np.mean(q_lats))
                rd["all_lats"].extend(q_lats)
                rd["rpcs"].append(np.mean(q_rpcs))
                rd["unique"].append(np.mean(q_unique))

    for ch in channels.values():
        ch.close()

    rows = []
    for ttl in TTL_VALUES:
        rd = run_data[ttl]
        n_all = len(rd["recalls_all"])
        if not n_all:
            continue
        avg_failure_rate    = np.mean(rd["failures"])    / NUM_QUERIES
        avg_timeout_rate    = np.mean(rd["timeouts"])    / NUM_QUERIES
        avg_incomplete_rate = np.mean(rd["incomplete"])  / NUM_QUERIES

        recalls_all    = rd["recalls_all"]
        avg_recall_all = np.mean(recalls_all) * 100
        if n_all >= 2:
            sem = np.std(recalls_all, ddof=1) * 100 / np.sqrt(n_all)
            tc  = _t_crit(n_all)
            ci_all_lo = max(0.0,   avg_recall_all - tc * sem)
            ci_all_hi = min(100.0, avg_recall_all + tc * sem)
        else:
            ci_all_lo = ci_all_hi = float("nan")

        run_recalls = rd["recalls"]
        n_succ = len(run_recalls)
        if n_succ > 0:
            avg_recall  = np.mean(run_recalls) * 100
            avg_lat     = np.mean(rd["latencies"])
            all_lats    = rd["all_lats"]
            avg_p50_lat = float(np.percentile(all_lats, 50))
            avg_p95_lat = float(np.percentile(all_lats, 95))
            avg_p99_lat = float(np.percentile(all_lats, 99))
            avg_rpcs    = np.mean(rd["rpcs"])
            avg_unique  = np.mean(rd["unique"])
            if n_succ >= 2:
                tc_s    = _t_crit(n_succ)
                sem_r   = np.std(run_recalls,      ddof=1) * 100 / np.sqrt(n_succ)
                sem_l   = np.std(rd["latencies"],  ddof=1)       / np.sqrt(n_succ)
                ci_lo     = max(0.0,   avg_recall - tc_s * sem_r)
                ci_hi     = min(100.0, avg_recall + tc_s * sem_r)
                lat_ci_lo = avg_lat - tc_s * sem_l
                lat_ci_hi = avg_lat + tc_s * sem_l
            else:
                ci_lo = ci_hi = lat_ci_lo = lat_ci_hi = float("nan")
        else:
            avg_recall = ci_lo = ci_hi = float("nan")
            avg_lat = lat_ci_lo = lat_ci_hi = float("nan")
            avg_p50_lat = avg_p95_lat = avg_p99_lat = avg_rpcs = avg_unique = float("nan")

        print(f"TTL={ttl}: "
              f"recall_all={avg_recall_all:.2f}% [{ci_all_lo:.2f}, {ci_all_hi:.2f}]  "
              f"recall_ok={avg_recall:.2f}%  "
              f"fail={avg_failure_rate:.1%}  timeout={avg_timeout_rate:.1%}  "
              f"incomplete={avg_incomplete_rate:.1%}  "
              f"{avg_lat:.1f} [{lat_ci_lo:.1f}, {lat_ci_hi:.1f}] ms  "
              f"p50={avg_p50_lat:.1f}  p95={avg_p95_lat:.1f}  p99={avg_p99_lat:.1f} ms  "
              f"rpcs={avg_rpcs:.0f}  unique={avg_unique:.1f}  "
              f"(n_all={n_all}, n_ok={n_succ})")
        rows.append({
            "ttl":                         ttl,
            "n_runs":                      n_all,
            "failure_rate":                round(avg_failure_rate, 6),
            "timeout_rate":                round(avg_timeout_rate, 6),
            "incomplete_rate":             round(avg_incomplete_rate, 6),
            "recall_over_all_mean":        round(avg_recall_all, 4),
            "recall_over_all_ci95_low":    round(ci_all_lo, 4),
            "recall_over_all_ci95_high":   round(ci_all_hi, 4),
            "recall_on_success_mean":      round(avg_recall, 4),
            "recall_on_success_ci95_low":  round(ci_lo, 4),
            "recall_on_success_ci95_high": round(ci_hi, 4),
            "latency_mean_ms":             round(avg_lat, 4),
            "latency_ci95_low":            round(lat_ci_lo, 4),
            "latency_ci95_high":           round(lat_ci_hi, 4),
            "latency_p50_ms":              round(avg_p50_lat, 4),
            "latency_p95_ms":              round(avg_p95_lat, 4),
            "latency_p99_ms":              round(avg_p99_lat, 4),
            "rpc_count_mean":              round(avg_rpcs, 2),
            "unique_nodes_mean":           round(avg_unique, 2),
        })
    return rows


if __name__ == "__main__":
    run_evaluation()
