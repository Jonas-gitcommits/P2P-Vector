import os
import sys
import csv
import time
import random
import hashlib
import pickle
import subprocess
import numpy as np

try:
    import redis
    from redis.cluster import RedisCluster, ClusterNode
except ImportError:
    print("[FEHLER] Python-Paket 'redis' fehlt: pip install redis --break-system-packages")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[FEHLER] Python-Paket 'faiss' fehlt (fuer Ground-Truth-Berechnung):")
    print("  pip install faiss-cpu --break-system-packages")
    sys.exit(1)

try:
    from scipy.stats import t as _t_dist
except ImportError:
    print("[FEHLER] Python-Paket 'scipy' fehlt: pip install scipy --break-system-packages")
    sys.exit(1)


TOTAL_VECTORS = 20000       

N_LIST         = [50, 200, 500]  
NUM_RUNS       = 20            
NUM_QUERIES    = 150           
HNSW_CONTROL_N = [500]   

K            = 3             
DATASET_NAME = "ir"          
DIM          = 384 if DATASET_NAME == "ir" else 128
SEED         = 1234           

BASE_PORT  = int(os.environ.get("BASE_PORT", "20000"))
HOST       = "127.0.0.1"
INDEX_NAME_FLAT = "idx_flat"
INDEX_NAME_HNSW = "idx_hnsw"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_RUNTIME = 10
KEY_PREFIX = "doc:"
HERE       = os.path.dirname(os.path.abspath(__file__))
OUT_CSV    = os.path.join(HERE, "results_redis.csv")
SETUP_SH   = os.path.join(HERE, "setup_cluster.sh")
TEARDOWN_SH = os.path.join(HERE, "teardown_cluster.sh")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def load_or_generate_data():
    ds_path = os.path.join(HERE, "dataset.npy")
    q_path = os.path.join(HERE, "queries.npy")

    if os.path.exists(ds_path) and os.path.exists(q_path):
        dataset = np.load(ds_path).astype(np.float32)
        queries = np.load(q_path).astype(np.float32)
        log(f"Echter Datensatz geladen: dataset={dataset.shape} queries={queries.shape}")
        if dataset.shape[1] != DIM:
            log(f"[WARNUNG] DIM in diesem Skript ({DIM}) != Spaltenzahl in dataset.npy "
                f"({dataset.shape[1]}). DIM anpassen!")
        return dataset, queries, True

    log("[WARNUNG] dataset.npy/queries.npy nicht gefunden neben diesem Skript.")
    log("[WARNUNG] Erzeuge synthetischen Ersatzdatensatz -- NUR Funktionspruefung, "
        "Ergebnisse NICHT fuer die Arbeit verwertbar.")
    rng = np.random.RandomState(SEED)
    n_synth = max(TOTAL_VECTORS, 500)
    dataset = rng.normal(size=(n_synth, DIM)).astype(np.float32)
    queries = rng.normal(size=(200, DIM)).astype(np.float32)
    return dataset, queries, False


def build_ground_truth_ids(dataset, queries, subset_size, k, dataset_name):
    h = hashlib.md5(dataset[:subset_size].tobytes() + queries.tobytes()).hexdigest()[:8]
    cache_file = os.path.join(HERE, f"gt_cache_{dataset_name}_size{subset_size}_k{k}_{h}.pkl")
    if os.path.exists(cache_file):
        log(f"Ground-Truth aus Cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    log(f"Berechne Ground-Truth mit FAISS (Cache: {cache_file}) ...")
    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset[:subset_size])
    _, indices = index.search(queries, k)
    true_ids = [set(indices[i].tolist()) for i in range(len(queries))]
    with open(cache_file, "wb") as f:
        pickle.dump(true_ids, f)
    return true_ids

def _stream_subprocess(cmd, env=None):
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    return proc.returncode


def setup_cluster(n):
    log(f"Setze Cluster mit {n} Shards auf (kann bei groesserem N ein paar Sekunden dauern) ...")
    env = os.environ.copy()
    env["BASE_PORT"] = str(BASE_PORT)
    returncode = _stream_subprocess(["bash", SETUP_SH, str(n)], env=env)
    return returncode == 0


def teardown_cluster():
    _stream_subprocess(["bash", TEARDOWN_SH])

CONNECT_TIMEOUT_S = float(os.environ.get("REDIS_CONNECT_TIMEOUT_S", "30"))
SOCKET_TIMEOUT_S = float(os.environ.get("REDIS_SOCKET_TIMEOUT_S", "30"))


def create_index(ports):
    flat_cmd = [
        "FT.CREATE", INDEX_NAME_FLAT,
        "ON", "HASH", "PREFIX", "1", KEY_PREFIX,
        "SCHEMA", "vec", "VECTOR", "FLAT", "6",
        "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2",
    ]
    hnsw_cmd = [
        "FT.CREATE", INDEX_NAME_HNSW,
        "ON", "HASH", "PREFIX", "1", KEY_PREFIX,
        "SCHEMA", "vec", "VECTOR", "HNSW", "10",
        "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2",
        "M", str(HNSW_M), "EF_CONSTRUCTION", str(HNSW_EF_CONSTRUCTION),
    ]
    
    first_port = ports[0]
    log(f"  Sende Index-Erstellung an Koordinator (Port {first_port}) ...")
    
    c = redis.Redis(host=HOST, port=first_port, decode_responses=False,
                     socket_connect_timeout=CONNECT_TIMEOUT_S,
                     socket_timeout=SOCKET_TIMEOUT_S)
    try:
        for cmd in (flat_cmd, hnsw_cmd):
            try:
                c.execute_command(*cmd)
            except redis.ResponseError as e:
                if "Index already exists" not in str(e):
                    raise
        log("  Indizes (FLAT+HNSW) erfolgreich im gesamten Cluster angelegt.")
    except Exception as e:
        log(f"[FEHLER] Index-Erstellung auf Port {first_port} fehlgeschlagen: {e}")
        raise
    finally:
        c.close()


def write_vectors(ports, dataset, total_vectors):
    startup_nodes = [ClusterNode(HOST, p) for p in ports]
    rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=False,
                       socket_connect_timeout=CONNECT_TIMEOUT_S,
                       socket_timeout=SOCKET_TIMEOUT_S)
    progress_step = max(total_vectors // 10, 1)
    for i in range(total_vectors):
        key = f"{KEY_PREFIX}{i}"
        rc.hset(key, mapping={
            "vec": dataset[i].astype(np.float32).tobytes(),
            "doc_id": str(i),
        })
        if (i + 1) % progress_step == 0 or (i + 1) == total_vectors:
            log(f"  {i + 1}/{total_vectors} Vektoren geschrieben ...")
    log(f"{total_vectors} Vektoren geschrieben (Keys {KEY_PREFIX}0 .. {KEY_PREFIX}{total_vectors-1})")


def _extract_doc_id_from_key(key):
    if isinstance(key, bytes):
        key = key.decode()
    return int(key[len(KEY_PREFIX):])


def parse_search_results(res):
    out = []

    def _to_str(x):
        return x.decode() if isinstance(x, bytes) else x

    if isinstance(res, dict):
        results = res.get(b"results", res.get("results", []))
        for item in results:
            key = item.get(b"id", item.get("id"))
            attrs = item.get(b"extra_attributes", item.get("extra_attributes", {})) or {}
            score = attrs.get(b"score", attrs.get("score"))
            if key is None or score is None:
                continue
            out.append((_extract_doc_id_from_key(key), float(_to_str(score))))
        return out

    if isinstance(res, (list, tuple)):
        pairs = res[1:] if res else []
        for i in range(0, len(pairs), 2):
            key = pairs[i]
            fields = pairs[i + 1]
            fd = {}
            for j in range(0, len(fields), 2):
                fd[fields[j]] = fields[j + 1]
            score = fd.get(b"score", fd.get("score"))
            if key is None or score is None:
                continue
            out.append((_extract_doc_id_from_key(key), float(_to_str(score))))
        return out

    return out


def coordinated_query(entry_client, index_name, query_bytes, k, ef_runtime=None):
    knn_clause = f"*=>[KNN {k} @vec $BLOB"
    if ef_runtime is not None:
        knn_clause += f" EF_RUNTIME {ef_runtime}"
    knn_clause += " AS score]"
    search_cmd = [
        "FT.SEARCH", index_name, knn_clause,
        "PARAMS", "2", "BLOB", query_bytes,
        "SORTBY", "score",
        "RETURN", "1", "score",
        "DIALECT", "2",
    ]
    t0 = time.time()
    res = entry_client.execute_command(*search_cmd)
    latency_ms = (time.time() - t0) * 1000
    candidates = parse_search_results(res)
    candidates.sort(key=lambda x: x[1])
    top_k_ids = [doc_id for doc_id, _ in candidates[:k]]
    return top_k_ids, latency_ms


def _fanout_snapshot(clients):
    total = 0
    for c in clients:
        stats = c.info("commandstats")
        total += stats.get("cmdstat__FT.SEARCH", {}).get("calls", 0)
    return total


def _ci95(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan")
    m = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    tc = float(_t_dist.ppf(0.975, n - 1))
    return m - tc * sem, m + tc * sem

def main():
    dataset, queries, is_real_data = load_or_generate_data()
    true_ids_all = build_ground_truth_ids(dataset, queries, TOTAL_VECTORS, K, DATASET_NAME)

    rows = []
    for n in N_LIST:
        log(f"=== N={n} Shards ===")
        setup_ok = False
        for setup_attempt in range(1, 3):
            setup_ok = setup_cluster(n)
            if setup_ok:
                break
            log(f"[WARNUNG] Setup fuer N={n} Versuch {setup_attempt}/2 fehlgeschlagen "
                f"-- raeume auf und versuche es {'noch einmal' if setup_attempt == 1 else 'nicht mehr'}.")
            teardown_cluster()
            if setup_attempt == 1:
                time.sleep(10)
        if not setup_ok:
            log(f"[FEHLER] Setup fuer N={n} auch im zweiten Versuch fehlgeschlagen -- ueberspringe.")
            continue

        try:
            ports = [BASE_PORT + i for i in range(n)]
            create_index(ports)
            write_vectors(ports, dataset, TOTAL_VECTORS)
            clients = [redis.Redis(host=HOST, port=p, decode_responses=False,
                                    socket_connect_timeout=CONNECT_TIMEOUT_S,
                                    socket_timeout=SOCKET_TIMEOUT_S) for p in ports]
            entry_client = clients[0] 

            algorithms_for_n = ["flat"] + (["hnsw"] if n in HNSW_CONTROL_N else [])
            for algo in algorithms_for_n:
                index_name = INDEX_NAME_FLAT if algo == "flat" else INDEX_NAME_HNSW
                ef_runtime = None if algo == "flat" else HNSW_EF_RUNTIME
                log(f"  --- Algorithmus: {algo} ---")

                recalls, latencies, rpc_counts = [], [], []
                for run_id in range(NUM_RUNS):
                    log(f"  Run {run_id + 1}/{NUM_RUNS} ({NUM_QUERIES} Anfragen) ...")
                    rng = random.Random(SEED + run_id)
                    idx = rng.sample(range(len(queries)), min(NUM_QUERIES, len(queries)))

                    fanout_before = _fanout_snapshot(clients)
                    for qi in idx:
                        query_bytes = queries[qi].astype(np.float32).tobytes()
                        top_k_ids, lat_ms = coordinated_query(
                            entry_client, index_name, query_bytes, K, ef_runtime)
                        matches = len(set(top_k_ids) & true_ids_all[qi])
                        recalls.append(matches / K)
                        latencies.append(lat_ms)
                    fanout_after = _fanout_snapshot(clients)
                    rpc_counts.append((fanout_after - fanout_before) / len(idx))

                recall_mean = float(np.mean(recalls)) * 100
                recall_ci_lo, recall_ci_hi = _ci95([r * 100 for r in recalls])
                lat_arr = np.array(latencies)
                rpc_mean = float(np.mean(rpc_counts))

                row = {
                    "dataset": DATASET_NAME + ("" if is_real_data else "(synthetic)"),
                    "algorithm": algo,
                    "n_shards": n,
                    "total_vectors": TOTAL_VECTORS,
                    "k": K,
                    "num_runs": NUM_RUNS,
                    "num_queries": NUM_QUERIES,
                    "hnsw_m": HNSW_M if algo == "hnsw" else "",
                    "hnsw_ef_construction": HNSW_EF_CONSTRUCTION if algo == "hnsw" else "",
                    "hnsw_ef_runtime": HNSW_EF_RUNTIME if algo == "hnsw" else "",
                    "recall_mean": round(recall_mean, 4),
                    "recall_ci95_low": round(recall_ci_lo, 4),
                    "recall_ci95_high": round(recall_ci_hi, 4),
                    "rpc_count_mean": round(rpc_mean, 4),
                    "latency_mean_ms": round(float(np.mean(lat_arr)), 4),
                    "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
                    "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
                    "latency_p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
                }
                log(f"N={n} [{algo}]: recall={recall_mean:.2f}% [{recall_ci_lo:.2f},{recall_ci_hi:.2f}]  "
                    f"rpc={rpc_mean:.2f}  latenz_mean={row['latency_mean_ms']:.1f}ms  "
                    f"p50={row['latency_p50_ms']:.1f}  p95={row['latency_p95_ms']:.1f}")
                rows.append(row)

        finally:
            teardown_cluster()

        if rows:
            with open(OUT_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            log(f"Zwischenstand gespeichert: {OUT_CSV}")

    log(f"Fertig. {len(rows)}/{len(N_LIST)} Shard-Groessen erfolgreich. -> {OUT_CSV}")

import atexit
import errno

LOCKFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".redis_measure.lock")


def _acquire_lock():
    fd = os.open(LOCKFILE, os.O_CREAT | os.O_RDWR)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError) as e:
        if isinstance(e, OSError) and e.errno in (errno.EACCES, errno.EAGAIN):
            os.close(fd)
            print(f"[FEHLER] Es laeuft bereits ein redis_measure.py-Prozess "
                  f"(Lock {LOCKFILE} ist belegt). Vor einem Neustart pruefen:")
            print("  ps aux | grep redis_measure | grep -v grep")
            print("  jobs -l")
            sys.exit(1)
    os.write(fd, str(os.getpid()).encode())
    atexit.register(os.close, fd)

if __name__ == "__main__":
    _acquire_lock()
    main()
