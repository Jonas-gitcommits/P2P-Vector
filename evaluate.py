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

from scipy.stats import t
def _t_crit(n):
    return float(t.ppf(0.975, n - 1))

def build_ground_truth_ids(dataset, queries):
    central_index = faiss.IndexFlatL2(DIMENSION)
    central_index.add(dataset[:SUBSET_SIZE])
    _, indices = central_index.search(queries, K)
    return [set(indices[i].tolist()) for i in range(len(queries))]


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
    print(f"  dataset: {dataset.shape}, queries verfügbar: {all_queries.shape}")
    print(f"  SUBSET_SIZE={SUBSET_SIZE}, early_stop_threshold={early_stop_threshold}, "
          f"EVAL_VARIANT={EVAL_VARIANT}, scenario={scenario_label}")

    print("Berechne Ground-Truth-IDs (alle Queries)...")
    true_ids_all = build_ground_truth_ids(dataset, all_queries)
    print(f"  Fertig: {len(true_ids_all)} Referenzmengen.\n")

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

    run_data = {ttl: {v: {"recalls": [], "latencies": [], "p95_lats": [],
                           "rpcs": [], "unique": [],
                           "failures": [], "timeouts": [], "recalls_all": []}
                      for v in variants}
                for ttl in TTL_VALUES}

    for run_id in range(NUM_RUNS):
        rng = random.Random(42 + run_id)
        idx = rng.sample(range(len(all_queries)), NUM_QUERIES)
        run_queries = all_queries[idx]
        run_true_ids = [true_ids_all[j] for j in idx]
        entry_nodes = [rng.choice(alive_nodes) for _ in range(NUM_QUERIES)]

        for ttl in TTL_VALUES:
            for variant in variants:
                print(f"Run {run_id + 1}/{NUM_RUNS} | TTL={ttl} | variant={variant} "
                      f"({NUM_QUERIES} Queries)...")

                q_recalls, q_lats, q_rpcs, q_unique = [], [], [], []
                q_failures, q_timeouts = 0, 0

                for i, query_vector in enumerate(run_queries):
                    request = make_request(query_vector, K, ttl, variant, fanout_k,
                                          early_stop_threshold)
                    try:
                        t0 = time.time()
                        response = stubs[entry_nodes[i]].SearchSimilar(request, timeout=10.0)
                        q_lats.append((time.time() - t0) * 1000)
                        matches = len(set(response.vector_ids[:K]) & run_true_ids[i])
                        q_recalls.append(matches / K)
                        q_rpcs.append(response.rpc_count)
                        q_unique.append(len(response.visited_nodes))
                    except grpc.RpcError as e:
                        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                            q_timeouts += 1
                        else:
                            q_failures += 1

                assert len(q_recalls) + q_failures + q_timeouts == NUM_QUERIES, (
                    f"Run {run_id + 1}, TTL={ttl}, {variant}: "
                    f"success={len(q_recalls)} failures={q_failures} timeouts={q_timeouts} "
                    f"!= NUM_QUERIES={NUM_QUERIES}"
                )

                rd = run_data[ttl][variant]
                rd["failures"].append(q_failures)
                rd["timeouts"].append(q_timeouts)
                rd["recalls_all"].append(sum(q_recalls) / NUM_QUERIES)

                if q_recalls:
                    rd["recalls"].append(np.mean(q_recalls))
                    rd["latencies"].append(np.mean(q_lats))
                    rd["p95_lats"].append(np.percentile(q_lats, 95))
                    rd["rpcs"].append(np.mean(q_rpcs))
                    rd["unique"].append(np.mean(q_unique))

    for ch in channels.values():
        ch.close()

    print()
    csv_rows = []

    for ttl in TTL_VALUES:
        for variant in variants:
            rd = run_data[ttl][variant]
            n_all = len(rd["recalls_all"])
            if not n_all:
                continue
            tc_all = _t_crit(n_all)

            avg_failure_rate = np.mean(rd["failures"]) / NUM_QUERIES
            avg_timeout_rate = np.mean(rd["timeouts"]) / NUM_QUERIES

            recalls_all = rd["recalls_all"]
            avg_recall_all = np.mean(recalls_all) * 100
            std_recall_all = np.std(recalls_all, ddof=1) * 100
            sem_recall_all = std_recall_all / np.sqrt(n_all)
            ci_all_lo = avg_recall_all - tc_all * sem_recall_all
            ci_all_hi = avg_recall_all + tc_all * sem_recall_all

            run_recalls = rd["recalls"]
            n_succ = len(run_recalls)
            if n_succ > 0:
                tc_succ = _t_crit(n_succ)
                avg_recall = np.mean(run_recalls) * 100
                std_recall = np.std(run_recalls, ddof=1) * 100
                sem_recall = std_recall / np.sqrt(n_succ)
                ci_lo = avg_recall - tc_succ * sem_recall
                ci_hi = avg_recall + tc_succ * sem_recall

                avg_lat = np.mean(rd["latencies"])
                std_lat = np.std(rd["latencies"], ddof=1)
                sem_lat = std_lat / np.sqrt(n_succ)
                lat_ci_lo = avg_lat - tc_succ * sem_lat
                lat_ci_hi = avg_lat + tc_succ * sem_lat
                avg_p95_lat = np.mean(rd["p95_lats"])
                avg_rpcs    = np.mean(rd["rpcs"])
                avg_unique  = np.mean(rd["unique"])
            else:
                avg_recall = ci_lo = ci_hi = std_recall = float("nan")
                avg_lat = lat_ci_lo = lat_ci_hi = std_lat = float("nan")
                avg_p95_lat = avg_rpcs = avg_unique = float("nan")

            print(f"TTL={ttl} [{variant}]: "
                  f"recall_all={avg_recall_all:.2f}% [{ci_all_lo:.2f}, {ci_all_hi:.2f}]  "
                  f"recall_ok={avg_recall:.2f}%  "
                  f"fail={avg_failure_rate:.1%}  timeout={avg_timeout_rate:.1%}  "
                  f"{avg_lat:.1f} [{lat_ci_lo:.1f}, {lat_ci_hi:.1f}] ms  "
                  f"p95={avg_p95_lat:.1f} ms  "
                  f"rpcs={avg_rpcs:.0f}  unique={avg_unique:.1f}  (n={n_all} Läufe)")
            csv_rows.append({
                "scenario":                    scenario_label,
                "ttl":                         ttl,
                "variant":                     variant,
                "n_runs":                      n_all,
                "failure_rate":                round(avg_failure_rate, 6),
                "timeout_rate":                round(avg_timeout_rate, 6),
                "recall_over_all_mean":        round(avg_recall_all, 4),
                "recall_over_all_ci95_low":    round(ci_all_lo, 4),
                "recall_over_all_ci95_high":   round(ci_all_hi, 4),
                "recall_on_success_mean":      round(avg_recall, 4),
                "recall_on_success_ci95_low":  round(ci_lo, 4),
                "recall_on_success_ci95_high": round(ci_hi, 4),
                "latency_mean_ms":             round(avg_lat, 4),
                "latency_std_ms":              round(std_lat, 4),
                "latency_ci95_low":            round(lat_ci_lo, 4),
                "latency_ci95_high":           round(lat_ci_hi, 4),
                "latency_p95_ms":              round(avg_p95_lat, 4),
                "rpc_count_mean":              round(avg_rpcs, 2),
                "unique_nodes_mean":           round(avg_unique, 2),
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
