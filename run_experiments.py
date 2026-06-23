import os, re, sys, csv, time, signal, subprocess, urllib.request, math
from datetime import datetime
import numpy as np
from scipy.stats import t as _t_dist
from evaluate import run_evaluation, run_topology

HERE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable

PROFILE    = "rebuild_bidir"
SMOKE_TEST = False

IR_CORPUS_SIZE = 200_000
BASE_SEED = 1234

EARLY_STOP_THRESHOLDS = {"sift": 65000.0, "ir": 0.90}
FAULT_PROFILE = dict(FAULT_KILL_INTERVAL=8.0, FAULT_KILL_PROBABILITY=0.4, FAULT_RESTART_DELAY=6.0)

PROFILES = {
    "standard": dict(
        DATASETS=["ir", "sift"],
        TOTAL_VECTORS=20000, N_BASE=20, N_LIST_SCALE=[10, 20, 30, 50, 100, 200, 500, 1000, 1500],
        NUM_QUERIES=300, NUM_QUERIES_LATENCY=150, NUM_RUNS=20,
        GOSSIP_WARMUP_S=30,
        TTL_CORE=[2, 4, 6], TTL_CHURN=[4, 6], TTL_LATENCY=[4],
        FANOUT_LIST=[1, 2, 3, 4], WARMUP_LIST=[0, 10, 30],
        CONNDROP_LIST=[0, 10, 30], FAULT_LEVELS=[0, 2, 4],
        NETWORK_BOOT_WAIT=6,
    ),
    "extra": dict(
        DATASETS=["ir"],
        TOTAL_VECTORS=20000, N_BASE=50,
        NUM_QUERIES=500, NUM_QUERIES_LATENCY=200, NUM_RUNS=20,
        GOSSIP_WARMUP_S=40, NETWORK_BOOT_WAIT=12,
        ADAPTIVE_ONLY=True,
        ITER_TTL_CONST=64,
       
        EF_SWEEP_N=[100, 200, 500, 1000], EF_SWEEP_VALUES=[16, 32, 48, 64, 96, 128],
        EF_SWEEP_RUNS=20,
       
        WARMUP_CROSS_N=[100, 200], WARMUP_CROSS_VALUES=[40, 120, 240],
        WARMUP_CROSS_EF=64, WARMUP_CROSS_RUNS=20,
        
        ALPHA_TEST_N=[100, 200], ALPHA_VALUES=[1, 3], ALPHA_EF=64, ALPHA_RUNS=20,
        
        SCALE_TTL_N=[200, 500, 1000], SCALE_TTL_VALUE=8, SCALE_TTL_EF=16, SCALE_TTL_RUNS=20,
        
        TOPOLOGY_N=[100, 200], TOPOLOGY_RUNS=20,
       
        CONNFIX_EF=64, CONNFIX_ALPHA=3, CONNFIX_TTL=64, CONNFIX_RUNS=8,
        CONNFIX_NQ_LARGE=150,
        CONNFIX_CELLS=[(100, 10), (200, 40), (500, 50), (1000, 60), (1500, 60)],
        CONNFIX_TTL_PROBE_VALUES=[6, 64],
    ),
    "rebuild_pilot": dict(
        DATASETS=["ir"], TOTAL_VECTORS=20000, N_BASE=500,
        NUM_QUERIES=300, NUM_QUERIES_LATENCY=200, NUM_RUNS=4,
        GOSSIP_WARMUP_S=60, NETWORK_BOOT_WAIT=12,
        REBUILD_ONLY=True,
        REBUILD_GRID_N=[500, 1000],
        REBUILD_EXPLORE_FRACTIONS=[0.667],
        REBUILD_COEFFS=[2.0],
        REBUILD_NQ_LARGE=150,
        REBUILD_ABLATION_N=[], REBUILD_ABLATION_ARMS=[],
        REBUILD_ABLATION_FRACTION=0.667, REBUILD_ABLATION_COEFF=2.0,
        REBUILD_OLDRPS_N=[500, 1000],
        REBUILD_RUNS_BY_N={500: 4, 1000: 4},
        REBUILD_EF_BASE=10, REBUILD_EF_PER_LOG2N=2,
    ),
    "rebuild_focus": dict(
        DATASETS=["ir"], TOTAL_VECTORS=20000, N_BASE=500,
        NUM_QUERIES=300, NUM_QUERIES_LATENCY=200, NUM_RUNS=5,
        GOSSIP_WARMUP_S=60, NETWORK_BOOT_WAIT=12, REBUILD_ONLY=True,
        REBUILD_SWEEP_N=[500, 1000],
        REBUILD_EXPLORE_FRACTIONS=[0.5, 0.667, 0.75],
        REBUILD_COEFFS=[1.0, 2.0],
        REBUILD_ANCHOR_N=[200, 1500],
        REBUILD_ANCHOR_FRACTIONS=[0.667],
        REBUILD_ANCHOR_COEFFS=[1.0, 2.0],
        REBUILD_NQ_LARGE=150,
        REBUILD_ABLATION_N=[500],
        REBUILD_ABLATION_ARMS=["stageb_anchor", "stageb", "with_far_bias", "no_shuffle", "no_bounded", "bidir"],
        REBUILD_ABLATION_FRACTION=0.667, REBUILD_ABLATION_COEFF=2.0,
        REBUILD_OLDRPS_N=[500, 1000],
        REBUILD_RUNS_BY_N={200: 8, 500: 5, 1000: 5, 1500: 3},
        REBUILD_EF_BASE=10, REBUILD_EF_PER_LOG2N=2,
    ),
    "rebuild_bidir": dict(
        DATASETS=["ir"], TOTAL_VECTORS=20000, N_BASE=500,
        NUM_QUERIES=300, NUM_QUERIES_LATENCY=200, NUM_RUNS=5,
        GOSSIP_WARMUP_S=60, NETWORK_BOOT_WAIT=12, REBUILD_ONLY=True,
        REBUILD_SWEEP_N=[],
        REBUILD_EXPLORE_FRACTIONS=[0.667],
        REBUILD_COEFFS=[2.0],
        REBUILD_ANCHOR_N=[500, 1000, 1500],
        REBUILD_ANCHOR_FRACTIONS=[0.667],
        REBUILD_ANCHOR_COEFFS=[2.0],
        REBUILD_GRID_FLAGS=dict(BOUNDED_VIEW=True, PEER_SHUFFLE=True,
                                BIDIRECTIONAL_NEIGHBORS=True,
                                FARTHEST_ANCHOR=False, FAR_RANDOM_BIAS=False),
        REBUILD_NQ_LARGE=150,
        REBUILD_ABLATION_N=[], REBUILD_ABLATION_ARMS=[],
        REBUILD_ABLATION_FRACTION=0.667, REBUILD_ABLATION_COEFF=2.0,
        REBUILD_OLDRPS_N=[],
        REBUILD_RUNS_BY_N={500: 5, 1000: 5, 1500: 3},
        REBUILD_EF_BASE=10, REBUILD_EF_PER_LOG2N=2,
    ),
}
P = PROFILES[PROFILE]
PORT_RELEASE_WAIT = 4

if SMOKE_TEST:
    P = dict(P)
    P.update(TOTAL_VECTORS=20000, N_LIST_SCALE=[10], NUM_QUERIES=20,
             NUM_QUERIES_LATENCY=20, NUM_RUNS=3, GOSSIP_WARMUP_S=30,
             NETWORK_BOOT_WAIT=6)

VAR_PLACEMENT = "PLACEMENT"
VAR_ROUTING   = "ROUTING_STRATEGY"
VAR_FANOUT    = "ROUTING_FANOUT"
VAR_WARMUP    = "GOSSIP_WARMUP_S"
VAR_CONNDROP  = "TOXIC_CONN_DROP_PCT"
VAR_ALPHA     = "ROUTING_ALPHA"
VAR_EF        = "ROUTING_EF"

OUT_CSV  = os.path.join(HERE, "experiment_results.csv")
CONFIG   = os.path.join(HERE, "config.py")
START = time.time()
_all_rows, _sim_proc, _tox_proc, _done, _total = [], None, None, 0, 0

TOXIPROXY_BIN = os.path.join(HERE, "toxiproxy-server")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

def elapsed():
    s = int(time.time() - START)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"

def _iter(routing):
    return {VAR_ALPHA: 3, VAR_EF: 16} if routing == "iterative" else {}

def _total_for(ds):
    return min(P["TOTAL_VECTORS"], IR_CORPUS_SIZE) if ds == "ir" else P["TOTAL_VECTORS"]

def _meta(block, placement, routing, n, vpn, fmd, repl, scen, dataset):
    return dict(block=block, placement=placement, routing=routing, num_nodes=n,
                vectors_per_node=vpn, fault_max_down=fmd, replication=repl,
                latency_scenario=scen, dataset=dataset)

def meta_str(m):
    extra = "".join(f" {k}={m[k]}" for k in ("fanout", "gossip_warmup_s", "conn_drop_pct", "early_stop", "routing_ef", "ef_rule", "ttl_mode", "routing_alpha", "fanout_rule", "ttl_rule", "bidirectional", "random_gossip", "explore_fraction", "max_neighbors_coeff", "arm") if k in m)
    return (f"{m['block']} ds={m['dataset']} place={m['placement']} route={m['routing']} "
            f"N={m['num_nodes']} fault={m['fault_max_down']} repl={m['replication']} "
            f"lat={m['latency_scenario']}{extra}")

def base_cfg(n, ttl, nq, total, **extra):
    cfg = {
        "NUM_NODES": n, "VECTORS_PER_NODE": total // n,
        "NUM_QUERIES": nq, "NUM_RUNS": 1, "TTL_VALUES": list(ttl),
        "GOSSIP_WARMUP_S": P["GOSSIP_WARMUP_S"], "TOXIPROXY_ENABLED": False,
        "LATENCY_SCENARIO": "none", "FAULT_INJECTION_ENABLED": False,
        "FAULT_MAX_DOWN": 0, "REPLICATION": True, "EARLY_STOP_ENABLED": False,
        VAR_PLACEMENT: "clustered", VAR_ROUTING: "greedy", VAR_FANOUT: 2,
        VAR_CONNDROP: 0.0, VAR_ALPHA: 3, VAR_EF: 16, "ROUTING_DEBUG": False,
        "BIDIRECTIONAL_NEIGHBORS": False, "RANDOM_GOSSIP": False,
        "BOUNDED_VIEW": False, "PEER_SHUFFLE": False, "FARTHEST_ANCHOR": False,
        "FAR_RANDOM_BIAS": False, "INCREMENTAL_START": False, "LOOP_JITTER": False,
        "FAILURE_STRIKES": 1, "MEASURE_TOPOLOGY": False,
        "EXPLORE_FRACTION": 0.5, "MAX_NEIGHBORS_COEFF": 1.0,
    }
    cfg.update(extra)
    return cfg

def set_config(**kv):
    with open(CONFIG) as f:
        src = f.read()
    for k, v in kv.items():
        pat = re.compile(rf"^{re.escape(k)}\s*=.*$", re.M)
        line = f"{k} = {v!r}"
        src = pat.sub(line, src) if pat.search(src) else src + f"\n{line}\n"
    with open(CONFIG, "w") as f:
        f.write(src)

def toxiproxy_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:8474/version", timeout=2)
        return True
    except Exception:
        return False

def start_toxiproxy():
    global _tox_proc
    if toxiproxy_up():
        return True
    if not os.path.exists(TOXIPROXY_BIN):
        log(f"toxiproxy-server nicht gefunden: {TOXIPROXY_BIN}")
        return False
    _tox_proc = subprocess.Popen(
        [TOXIPROXY_BIN, "-host", "127.0.0.1", "-port", "8474"],
        cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(20):
        time.sleep(0.3)
        if toxiproxy_up():
            log("Toxiproxy gestartet.")
            return True
    log("Toxiproxy konnte nicht gestartet werden.")
    return False

def stop_toxiproxy():
    global _tox_proc
    if _tox_proc is None:
        return
    try:
        pgid = os.getpgid(_tox_proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        _tox_proc.wait(timeout=5)
    except Exception:
        pass
    _tox_proc = None

def start_network(n):
    global _sim_proc
    subprocess.run(["pkill", "-f", "node\\.py.*P2PVEC_MARKER"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    _sim_proc = subprocess.Popen([PY, "simulator.py"], cwd=HERE, start_new_session=True)
    time.sleep(P["NETWORK_BOOT_WAIT"] + 0.12 * n)

def stop_network():
    global _sim_proc
    if _sim_proc is None:
        return
    try:
        pgid = os.getpgid(_sim_proc.pid)
        os.killpg(pgid, signal.SIGINT)
        _sim_proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    _sim_proc = None
    subprocess.run(["pkill", "-f", "node\\.py.*P2PVEC_MARKER"], stderr=subprocess.DEVNULL)
    time.sleep(PORT_RELEASE_WAIT)

def generate_data():
    log("  Daten neu erzeugen ...")
    try:
        r = subprocess.run([PY, "generate_data.py"], cwd=HERE, timeout=900,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            log("  generate_data fehlgeschlagen:\n" + r.stdout[-800:])
            return False
    except subprocess.TimeoutExpired:
        log("  generate_data Timeout.")
        return False
    return True

def save():
    if not _all_rows:
        return
    meta_first = ["dataset", "block", "placement", "routing", "num_nodes", "vectors_per_node",
                  "fault_max_down", "replication", "latency_scenario",
                  "fanout", "gossip_warmup_s", "conn_drop_pct", "early_stop"]
    keys = list(dict.fromkeys(k for r in _all_rows for k in r))
    ordered = [k for k in meta_first if k in keys] + [k for k in keys if k not in meta_first]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        w.writerows(_all_rows)

def _t_ci(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan")
    m = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    tc = float(_t_dist.ppf(0.975, n - 1))
    return m - tc * sem, m + tc * sem


def _aggregate(run_rows):
    if not run_rows:
        return []
    by_ttl = {}
    for row in run_rows:
        by_ttl.setdefault(row["ttl"], []).append(row)

    def _mean(rows, key):
        return round(float(np.nanmean([float(r[key]) for r in rows])), 4)

    def _topo_mean(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    def _ci(vals, lo_bound=None, hi_bound=None):
        lo, hi = _t_ci(vals)
        if lo_bound is not None and not np.isnan(lo):
            lo = max(lo_bound, lo)
        if hi_bound is not None and not np.isnan(hi):
            hi = min(hi_bound, hi)
        return round(lo, 4), round(hi, 4)

    result = []
    for ttl in sorted(by_ttl, key=lambda x: int(x)):
        rows = by_ttl[ttl]
        recalls    = [float(r["recall_over_all_mean"])   for r in rows]
        recalls_ok = [float(r["recall_on_success_mean"]) for r in rows]
        lats       = [float(r["latency_mean_ms"])        for r in rows]
        rci_lo,  rci_hi  = _ci(recalls,    0.0, 100.0)
        roci_lo, roci_hi = _ci(recalls_ok, 0.0, 100.0)
        lci_lo,  lci_hi  = _ci(lats)
        alive_vals = [float(r["alive_count"]) for r in rows if r.get("alive_count") is not None]
        wait_vals  = [float(r["ready_wait_s"]) for r in rows if r.get("ready_wait_s") is not None]
        nbm_vals   = [float(r["neighbor_count_mean"]) for r in rows if r.get("neighbor_count_mean") is not None]
        nbmin_vals = [float(r["neighbor_count_min"])  for r in rows if r.get("neighbor_count_min")  is not None]
        nbmax_vals = [float(r["neighbor_count_max"])  for r in rows if r.get("neighbor_count_max")  is not None]
        lat_pool = []
        for r in rows:
            lat_pool.extend(r.get("_lat_samples") or [])
        if lat_pool:
            p50 = round(float(np.percentile(lat_pool, 50)), 4)
            p95 = round(float(np.percentile(lat_pool, 95)), 4)
            p99 = round(float(np.percentile(lat_pool, 99)), 4)
        else:
            p50 = _mean(rows, "latency_p50_ms")
            p95 = _mean(rows, "latency_p95_ms")
            p99 = _mean(rows, "latency_p99_ms")
        result.append({
            "ttl":                         ttl,
            "n_runs":                      len(rows),
            "failure_rate":                _mean(rows, "failure_rate"),
            "timeout_rate":                _mean(rows, "timeout_rate"),
            "recall_over_all_mean":        round(float(np.mean(recalls)), 4),
            "recall_over_all_ci95_low":    rci_lo,
            "recall_over_all_ci95_high":   rci_hi,
            "recall_on_success_mean":      round(float(np.nanmean(recalls_ok)), 4),
            "recall_on_success_ci95_low":  roci_lo,
            "recall_on_success_ci95_high": roci_hi,
            "latency_mean_ms":             round(float(np.mean(lats)), 4),
            "latency_ci95_low":            lci_lo,
            "latency_ci95_high":           lci_hi,
            "latency_p50_ms":              p50,
            "latency_p95_ms":              p95,
            "latency_p99_ms":              p99,
            "rpc_count_mean":              round(float(np.nanmean([float(r["rpc_count_mean"])    for r in rows])), 2),
            "unique_nodes_mean":           round(float(np.nanmean([float(r["unique_nodes_mean"]) for r in rows])), 2),
            "alive_count_min":             int(min(alive_vals)) if alive_vals else None,
            "ready_wait_s_max":            round(max(wait_vals), 1) if wait_vals else None,
            "neighbor_count_mean":         round(float(np.mean(nbm_vals)), 2) if nbm_vals else None,
            "neighbor_count_min":          int(min(nbmin_vals)) if nbmin_vals else None,
            "neighbor_count_max":          int(max(nbmax_vals)) if nbmax_vals else None,
            "reach_mean":                  _topo_mean(rows, "reach_mean"),
            "reach_min":                   _topo_mean(rows, "reach_min"),
            "reach_sym":                   _topo_mean(rows, "reach_sym"),
            "largest_scc_frac":            _topo_mean(rows, "largest_scc_frac"),
            "num_scc":                     _topo_mean(rows, "num_scc"),
            "indeg_mean":                  _topo_mean(rows, "indeg_mean"),
            "indeg_max":                   _topo_mean(rows, "indeg_max"),
            "cross_cluster_frac":          _topo_mean(rows, "cross_cluster_frac"),
        })
    return result


def _aggregate_topology(run_rows):
    rows = [r for r in run_rows if r]
    if not rows:
        return []
    def _mean(key):
        return round(float(np.mean([r[key] for r in rows])), 3)
    return [{
        "n_runs":                len(rows),
        "deg_min":               int(min(r["deg_min"] for r in rows)),
        "deg_mean":              _mean("deg_mean"),
        "deg_max":               int(max(r["deg_max"] for r in rows)),
        "reach_directed_mean":   _mean("reach_directed"),
        "largest_scc_mean":      _mean("largest_scc"),
        "reach_sym_mean":        _mean("reach_sym"),
        "search_visited_mean":   _mean("search_visited"),
        "search_reachable_mean": _mean("search_reachable"),
        "search_overlap_mean":   _mean("search_overlap"),
        "alive_count_min":       int(min(r["alive_count"] for r in rows)),
    }]


def run_condition(meta, cfg, regen):
    global _done
    _done += 1
    log(f"[{_done}/{_total}] {meta_str(meta)}  ({elapsed()})")
    set_config(**cfg)
    if regen and not generate_data():
        return False
    measure  = meta.pop("measure", "search")
    num_runs = meta.pop("num_runs", P["NUM_RUNS"])
    run_rows = []
    for r in range(num_runs):
        set_config(SEED=BASE_SEED + r)
        try:
            start_network(meta["num_nodes"])
            run_rows += run_topology() if measure == "topology" else run_evaluation()
        except Exception as e:
            log(f"  Fehler: {e!r}")
        finally:
            stop_network()
    rows = _aggregate_topology(run_rows) if measure == "topology" else _aggregate(run_rows)
    for row in rows:
        row.update(meta)
        _all_rows.append(row)
    save()
    log(f"  -> {len(rows)} Zeilen, gesamt {len(_all_rows)}.")
    return True

def _adaptive_iter_cfg(ef):
    return {VAR_ROUTING: "iterative", VAR_ALPHA: 3, VAR_EF: ef}

def build_adaptive_plan():
    plan = []
    ttl_const = P.get("ITER_TTL_CONST", 64)
    for ds in P.get("DATASETS", ["ir"]):
        total = _total_for(ds)

        for n in P.get("EF_SWEEP_N", []):
            for ef in P.get("EF_SWEEP_VALUES", []):
                m = _meta("ef_sweep", "clustered", "iterative", n, total // n, 0, True, "none", ds)
                m["routing_ef"] = ef; m["ttl_mode"] = f"fixed{ttl_const}"
                m["num_runs"] = P.get("EF_SWEEP_RUNS", P["NUM_RUNS"])
                c = base_cfg(n, [ttl_const], P["NUM_QUERIES"], total,
                             DATASET=ds, **_adaptive_iter_cfg(ef))
                plan.append((m, c, False))

        ef_warm = P.get("WARMUP_CROSS_EF", 64)
        for n in P.get("WARMUP_CROSS_N", []):
            for w in P.get("WARMUP_CROSS_VALUES", []):
                m = _meta("warmup", "clustered", "iterative", n, total // n, 0, True, "none", ds)
                m["routing_ef"] = ef_warm; m["ttl_mode"] = f"fixed{ttl_const}"; m["gossip_warmup_s"] = w
                m["num_runs"] = P.get("WARMUP_CROSS_RUNS", P["NUM_RUNS"])
                c = base_cfg(n, [ttl_const], P["NUM_QUERIES"], total, DATASET=ds,
                             **{VAR_ROUTING: "iterative", VAR_ALPHA: 3, VAR_EF: ef_warm, VAR_WARMUP: w})
                plan.append((m, c, False))

        ef_alpha = P.get("ALPHA_EF", 64)
        for n in P.get("ALPHA_TEST_N", []):
            for a in P.get("ALPHA_VALUES", []):
                m = _meta("alpha", "clustered", "iterative", n, total // n, 0, True, "none", ds)
                m["routing_ef"] = ef_alpha; m["ttl_mode"] = f"fixed{ttl_const}"; m["routing_alpha"] = a
                m["num_runs"] = P.get("ALPHA_RUNS", P["NUM_RUNS"])
                c = base_cfg(n, [ttl_const], P["NUM_QUERIES"], total,
                             DATASET=ds, **{VAR_ROUTING: "iterative", VAR_ALPHA: a, VAR_EF: ef_alpha})
                plan.append((m, c, False))

        ttl_scale = P.get("SCALE_TTL_VALUE", 8)
        ef_scale  = P.get("SCALE_TTL_EF", 16)
        for n in P.get("SCALE_TTL_N", []):
            m = _meta("scale_ttl", "clustered", "iterative", n, total // n, 0, True, "none", ds)
            m["routing_ef"] = ef_scale; m["ttl_mode"] = f"fixed{ttl_scale}"
            m["num_runs"] = P.get("SCALE_TTL_RUNS", P["NUM_RUNS"])
            c = base_cfg(n, [ttl_scale], P["NUM_QUERIES"], total,
                         DATASET=ds, **_adaptive_iter_cfg(ef_scale))
            plan.append((m, c, False))

        for n in P.get("TOPOLOGY_N", []):
            m = _meta("topology", "clustered", "none", n, total // n, 0, True, "none", ds)
            m["measure"] = "topology"
            m["num_runs"] = P.get("TOPOLOGY_RUNS", P["NUM_RUNS"])
            c = base_cfg(n, [ttl_const], P["NUM_QUERIES"], total, DATASET=ds,
                         **_adaptive_iter_cfg(P.get("CONNFIX_EF", 64)))
            plan.append((m, c, False))

        cf_ef     = P.get("CONNFIX_EF", 64)
        cf_alpha  = P.get("CONNFIX_ALPHA", 3)
        cf_ttl    = P.get("CONNFIX_TTL", 64)
        cf_runs   = P.get("CONNFIX_RUNS", 8)
        cf_nq_lrg = P.get("CONNFIX_NQ_LARGE", P["NUM_QUERIES"])
        cf_cells  = P.get("CONNFIX_CELLS", [])
        combos    = [(False, False), (True, False), (False, True), (True, True)]

        def _connfix_cell(block, n, warmup, bidir, rgossip, ttl_v):
            nq = P["NUM_QUERIES"] if n <= 200 else cf_nq_lrg
            m = _meta(block, "clustered", "iterative", n, total // n, 0, True, "none", ds)
            m["routing_ef"] = cf_ef; m["routing_alpha"] = cf_alpha
            m["ttl_mode"] = f"fixed{ttl_v}"; m["gossip_warmup_s"] = warmup
            m["bidirectional"] = bidir; m["random_gossip"] = rgossip
            m["num_runs"] = cf_runs
            c = base_cfg(n, [ttl_v], nq, total, DATASET=ds,
                         **{VAR_ROUTING: "iterative", VAR_ALPHA: cf_alpha, VAR_EF: cf_ef, VAR_WARMUP: warmup},
                         BIDIRECTIONAL_NEIGHBORS=bidir, RANDOM_GOSSIP=rgossip)
            return (m, c, False)

        for (n, warmup) in cf_cells:
            for (bidir, rgossip) in combos:
                plan.append(_connfix_cell("connfix", n, warmup, bidir, rgossip, cf_ttl))

        for ttl_v in P.get("CONNFIX_TTL_PROBE_VALUES", [6, 64]):
            for (n, warmup) in cf_cells:
                plan.append(_connfix_cell("connfix_ttl_probe", n, warmup, True, True, ttl_v))

    return plan


def build_rebuild_plan():
    plan = []
    ds = "ir"
    total = _total_for(ds)
    common = dict(INCREMENTAL_START=True, LOOP_JITTER=True, FAILURE_STRIKES=3, MEASURE_TOPOLOGY=True)

    def _ef(n):
        return round(P["REBUILD_EF_BASE"] + P["REBUILD_EF_PER_LOG2N"] * math.log2(n))

    def _runs(n):
        return P["REBUILD_RUNS_BY_N"].get(n, P["NUM_RUNS"])

    def _nq(n):
        return P["NUM_QUERIES"] if n <= 200 else P["REBUILD_NQ_LARGE"]

    def _cell(block, n, design, use_common=True, arm=None, frac=None, coeff=None):
        ef = _ef(n)
        m = _meta(block, "clustered", "iterative", n, total // n, 0, True, "none", ds)
        m["routing_ef"] = ef
        m["ttl_mode"] = f"fixed{ef}"
        m["num_runs"] = _runs(n)
        if frac is not None:
            m["explore_fraction"] = frac
        if coeff is not None:
            m["max_neighbors_coeff"] = coeff
        if arm is not None:
            m["arm"] = arm
        
        c = base_cfg(n, [ef], _nq(n), total, DATASET=ds,
                     **{VAR_ROUTING: "iterative", VAR_ALPHA: 3, VAR_EF: ef})
        if use_common:
            c.update(common)
        c.update(design)
        return (m, c, False)

    

    grid_flags = P.get("REBUILD_GRID_FLAGS",
                       dict(BOUNDED_VIEW=True, PEER_SHUFFLE=True,
                            FARTHEST_ANCHOR=True, FAR_RANDOM_BIAS=False))

    for n in sorted(P.get("REBUILD_SWEEP_N", P.get("REBUILD_GRID_N", []))):
        for frac in P["REBUILD_EXPLORE_FRACTIONS"]:
            for coeff in P["REBUILD_COEFFS"]:
                design = dict(grid_flags,
                              EXPLORE_FRACTION=frac, MAX_NEIGHBORS_COEFF=coeff)
                plan.append(_cell("rebuild_grid", n, design, frac=frac, coeff=coeff))


    for n in sorted(P.get("REBUILD_ANCHOR_N", [])):
        for frac in P.get("REBUILD_ANCHOR_FRACTIONS", []):
            for coeff in P.get("REBUILD_ANCHOR_COEFFS", []):
                design = dict(grid_flags,
                              EXPLORE_FRACTION=frac, MAX_NEIGHBORS_COEFF=coeff)
                plan.append(_cell("rebuild_grid", n, design, frac=frac, coeff=coeff))

    
    af, ac = P["REBUILD_ABLATION_FRACTION"], P["REBUILD_ABLATION_COEFF"]
    arm_flags = {
        "stageb_anchor": dict(BOUNDED_VIEW=True,  PEER_SHUFFLE=True,  FARTHEST_ANCHOR=True,  FAR_RANDOM_BIAS=False),
        "stageb":        dict(BOUNDED_VIEW=True,  PEER_SHUFFLE=True,  FARTHEST_ANCHOR=False, FAR_RANDOM_BIAS=False),
        "with_far_bias": dict(BOUNDED_VIEW=True,  PEER_SHUFFLE=True,  FARTHEST_ANCHOR=True,  FAR_RANDOM_BIAS=True),
        "no_shuffle":    dict(BOUNDED_VIEW=True,  PEER_SHUFFLE=False, FARTHEST_ANCHOR=True,  FAR_RANDOM_BIAS=False),
        "no_bounded":    dict(BOUNDED_VIEW=False, PEER_SHUFFLE=True,  FARTHEST_ANCHOR=False, FAR_RANDOM_BIAS=False),
        "bidir":         dict(BOUNDED_VIEW=True,  PEER_SHUFFLE=True,  FARTHEST_ANCHOR=True,  FAR_RANDOM_BIAS=False, BIDIRECTIONAL_NEIGHBORS=True),
    }
    ablation_n = P.get("REBUILD_ABLATION_N", [])
    ablation_arms = P.get("REBUILD_ABLATION_ARMS", [])
    if ablation_n and ablation_arms:
        for n in ablation_n:
            for arm in ablation_arms:
                design = dict(arm_flags[arm], EXPLORE_FRACTION=af, MAX_NEIGHBORS_COEFF=ac)
                plan.append(_cell("rebuild_ablation", n, design, arm=arm))

    for n in P.get("REBUILD_OLDRPS_N", []):
        design = dict(BOUNDED_VIEW=False, PEER_SHUFFLE=False, FARTHEST_ANCHOR=False, FAR_RANDOM_BIAS=False,
                      RANDOM_GOSSIP=True, BIDIRECTIONAL_NEIGHBORS=True,
                      INCREMENTAL_START=False, LOOP_JITTER=False, FAILURE_STRIKES=1,
                      MEASURE_TOPOLOGY=True)
        plan.append(_cell("rebuild_oldrps", n, design, use_common=False, arm="old_rps"))

    return plan


def _block_active(name):
    sel = P.get("RUN_BLOCKS")
    return name in sel if sel else True


def build_plan():
    plan = []
    datasets = P.get("DATASETS", ["sift"])

    if SMOKE_TEST:
        for di, ds in enumerate(datasets):
            total = _total_for(ds)
            for routing in ["greedy", "flood", "iterative"]:
                m = _meta("SMOKE", "clustered", routing, 10, total // 10, 0, True, "none", ds)
                c = base_cfg(10, [2, 4], P["NUM_QUERIES"], total, DATASET=ds,
                             **{VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, False))
            for es in [False, True]:
                m = _meta("SMOKE_earlystop", "clustered", "greedy", 10, total // 10, 0, True, "none", ds)
                m["early_stop"] = es
                c = base_cfg(10, [4], P["NUM_QUERIES"], total, DATASET=ds,
                             EARLY_STOP_ENABLED=es,
                             EARLY_STOP_THRESHOLD=EARLY_STOP_THRESHOLDS.get(ds, 0.0))
                plan.append((m, c, False))
            if di == 0:
                m = _meta("SMOKE_churn", "clustered", "greedy", 10, total // 10, 2, True, "none", ds)
                c = base_cfg(10, [4], P["NUM_QUERIES"], total, DATASET=ds,
                             FAULT_INJECTION_ENABLED=True, FAULT_MAX_DOWN=2, **FAULT_PROFILE)
                plan.append((m, c, False))
                m = _meta("SMOKE_latency", "clustered", "greedy", 10, total // 10, 0, True, "mid", ds)
                c = base_cfg(10, [4], P["NUM_QUERIES_LATENCY"], total, DATASET=ds,
                             TOXIPROXY_ENABLED=True, LATENCY_SCENARIO="mid")
                plan.append((m, c, True))
        return plan

    if P.get("REBUILD_ONLY", False):
        return build_rebuild_plan()

    if P.get("ADAPTIVE_ONLY", False):
        return build_adaptive_plan()

    for ds in datasets:
        total = _total_for(ds)
        nb, vb = P["N_BASE"], total // P["N_BASE"]

        for placement in (["clustered", "contiguous"] if _block_active("ablation") else []):
            for routing in ["greedy", "random", "flood", "iterative"]:
                m = _meta("ablation", placement, routing, nb, vb, 0, True, "none", ds)
                c = base_cfg(nb, P["TTL_CORE"], P["NUM_QUERIES"], total, DATASET=ds,
                             **{VAR_PLACEMENT: placement, VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, False))

        for n in (P["N_LIST_SCALE"] if not P.get("SKIP_SCALE", False) and _block_active("scale") else []):
            heavy = n <= 200
            ttls  = P["TTL_CORE"] if heavy else [6]
            nq    = P["NUM_QUERIES"] if heavy else 100
            for routing in ["greedy", "flood", "iterative"]:
                m = _meta("scale", "clustered", routing, n, total // n, 0, True, "none", ds)
                m["num_runs"] = 5 if heavy else 3
                c = base_cfg(n, ttls, nq, total, DATASET=ds,
                             **{VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, False))

        fmax = max(P["FAULT_LEVELS"]) or 4
        for routing in (["greedy", "iterative"] if _block_active("churn") else []):
            for fmd in P["FAULT_LEVELS"]:
                m = _meta("churn", "clustered", routing, nb, vb, fmd, True, "none", ds)
                c = base_cfg(nb, P["TTL_CHURN"], P["NUM_QUERIES"], total, DATASET=ds,
                             FAULT_INJECTION_ENABLED=(fmd > 0), FAULT_MAX_DOWN=fmd,
                             **FAULT_PROFILE, **{VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, False))
            m = _meta("churn_repl_off", "clustered", routing, nb, vb, fmax, False, "none", ds)
            c = base_cfg(nb, P["TTL_CHURN"], P["NUM_QUERIES"], total, DATASET=ds,
                         REPLICATION=False, FAULT_INJECTION_ENABLED=True,
                         FAULT_MAX_DOWN=fmax, **FAULT_PROFILE,
                         **{VAR_ROUTING: routing}, **_iter(routing))
            plan.append((m, c, False))

        for scen in (["none", "mid", "high"] if _block_active("latency") else []):
            for routing in ["greedy", "flood", "iterative"]:
                m = _meta("latency", "clustered", routing, nb, vb, 0, True, scen, ds)
                c = base_cfg(nb, P["TTL_LATENCY"], P["NUM_QUERIES_LATENCY"], total,
                             DATASET=ds, TOXIPROXY_ENABLED=True, LATENCY_SCENARIO=scen,
                             **{VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, True))

        for fan in (P["FANOUT_LIST"] if _block_active("fanout") else []):
            m = _meta("fanout", "clustered", "greedy", nb, vb, 0, True, "none", ds)
            m["fanout"] = fan
            c = base_cfg(nb, P["TTL_CORE"], P["NUM_QUERIES"], total,
                         DATASET=ds, **{VAR_FANOUT: fan})
            plan.append((m, c, False))

        for w in (P["WARMUP_LIST"] if _block_active("gossip") else []):
            m = _meta("gossip", "clustered", "greedy", nb, vb, 0, True, "none", ds)
            m["gossip_warmup_s"] = w
            c = base_cfg(nb, [4], P["NUM_QUERIES"], total, DATASET=ds, **{VAR_WARMUP: w})
            plan.append((m, c, False))

        for pct in (P["CONNDROP_LIST"] if _block_active("conndrop") else []):
            m = _meta("conndrop", "clustered", "greedy", nb, vb, 0, True, "none", ds)
            m["conn_drop_pct"] = pct
            c = base_cfg(nb, [4], P["NUM_QUERIES_LATENCY"], total,
                         DATASET=ds, TOXIPROXY_ENABLED=True, **{VAR_CONNDROP: float(pct)})
            plan.append((m, c, True))

        ttl_h = P["TTL_CORE"][len(P["TTL_CORE"]) // 2:]
        for routing in (["greedy", "iterative"] if _block_active("earlystop") else []):
            for es in [False, True]:
                m = _meta("earlystop", "clustered", routing, nb, vb, 0, True, "none", ds)
                m["early_stop"] = es
                c = base_cfg(nb, ttl_h, P["NUM_QUERIES"], total, DATASET=ds,
                             EARLY_STOP_ENABLED=es,
                             EARLY_STOP_THRESHOLD=EARLY_STOP_THRESHOLDS.get(ds, 0.0),
                             **{VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, False))

    return plan

def _run_plan(plan):
    last_n, last_ds, skipped = None, None, []
    for meta, cfg, needs_tox in plan:
        if needs_tox and not toxiproxy_up():
            log(f"[{meta['block']}] uebersprungen (Toxiproxy nicht da).")
            skipped.append((meta, cfg, needs_tox))
            continue
        regen = meta["num_nodes"] != last_n or meta["dataset"] != last_ds
        if run_condition(meta, cfg, regen):
            last_n, last_ds = meta["num_nodes"], meta["dataset"]
    return skipped

def main():
    global _total
    for fn in ("config.py", "simulator.py", "evaluate.py", "generate_data.py"):
        if not os.path.exists(os.path.join(HERE, fn)):
            sys.exit(f"{fn} fehlt in {HERE}.")
    signal.signal(signal.SIGINT,  lambda s, f: (stop_network(), stop_toxiproxy(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda s, f: (stop_network(), stop_toxiproxy(), sys.exit(1)))

    if not P.get("ADAPTIVE_ONLY", False) and not P.get("REBUILD_ONLY", False):
        start_toxiproxy()

    plan = build_plan()
    _total = len(plan)
    log(f"Profil={PROFILE} SMOKE={SMOKE_TEST}  {_total} Bedingungen  "
        f"Toxiproxy {'ok' if toxiproxy_up() else 'NICHT erreichbar'}.")

    skipped = _run_plan(plan)

    if skipped and toxiproxy_up():
        log(f"Toxiproxy jetzt erreichbar -> {len(skipped)} Bedingungen nachholen.")
        _run_plan(skipped)
    elif skipped:
        log(f"{len(skipped)} Toxiproxy-Bedingungen ausgelassen.")

    log(f"FERTIG. {len(_all_rows)} Zeilen in {os.path.basename(OUT_CSV)}. {elapsed()}")

if __name__ == "__main__":
    try:
        main()
    finally:
        stop_network()
        stop_toxiproxy()