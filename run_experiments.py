import os, re, sys, csv, time, signal, subprocess, urllib.request, math
from datetime import datetime
import numpy as np
from scipy.stats import t as _t_dist
from evaluate import run_evaluation
from config import RPC_TIMEOUT_BASE_S

HERE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable

PROFILE    = "final_overlay"

IR_CORPUS_SIZE = 200_000
BASE_SEED = 1234

PROFILES = {
    "final_overlay": dict(
        TOTAL_VECTORS=20000,
        NUM_QUERIES=300, NUM_QUERIES_LARGE=150, NUM_RUNS=20,
        GOSSIP_WARMUP_S=60, NETWORK_BOOT_WAIT=12,
        OVERLAY_ONLY=True,
        TIER1=20, TIER2=10,
        FINAL_ACTIVE_BLOCKS=[1, 2, 3, 4, 5, 6],
        FINAL_ACTIVE_N=[200, 500, 750, 1000, 1250],
    ),
}
P = PROFILES[PROFILE]
PORT_RELEASE_WAIT = 4

RUN = "routing"  

_ALL_BLOCKS = [1, 2, 3, 4, 5, 6]
_ALL_N      = [50, 200, 500, 750, 1000, 1250]

RUN_AXES = {
    "routing":    ([1], [50],       "results_routing.csv"),
    "ablation":   ([2], [500],           "results_ablation.csv"),
    "parameter":  ([3], [500],           "results_parameter.csv"),
    "seeds":      ([4], _ALL_N,          "results_seeds.csv"),
    "scaling":    ([5], [500, 750, 1000, 1250], "results_scaling.csv"),
    "robustheit": ([6], [500],           "results_robustheit.csv"),
    "all":        (_ALL_BLOCKS, _ALL_N,  "results_all.csv"),
}
_RUN_BLOCKS, _RUN_N, _RUN_CSV_NAME = RUN_AXES[RUN]
P["FINAL_ACTIVE_BLOCKS"] = _RUN_BLOCKS
P["FINAL_ACTIVE_N"]      = _RUN_N

VAR_PLACEMENT = "PLACEMENT"
VAR_ROUTING   = "ROUTING_STRATEGY"
VAR_FANOUT    = "ROUTING_FANOUT"
VAR_CONNDROP  = "TOXIC_CONN_DROP_PCT"
VAR_ALPHA     = "ROUTING_ALPHA"
VAR_EF        = "ROUTING_EF"

OUT_CSV  = os.path.join(HERE, _RUN_CSV_NAME)
CONFIG   = os.path.join(HERE, "config.py")
START = time.time()
_all_rows, _sim_proc, _tox_proc, _done, _total = [], None, None, 0, 0

TOXIPROXY_BIN = os.path.join(HERE, "toxiproxy-server")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

def elapsed():
    s = int(time.time() - START)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"

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
        "NUM_NODES": n, "NUM_SEED_NODES": 5, "VECTORS_PER_NODE": total // n,
        "NUM_QUERIES": nq, "NUM_RUNS": 1, "TTL_VALUES": list(ttl),
        "GOSSIP_WARMUP_S": P["GOSSIP_WARMUP_S"], "TOXIPROXY_ENABLED": False,
        "LATENCY_SCENARIO": "none", "FAULT_INJECTION_ENABLED": False,
        "FAULT_MAX_DOWN": 0, "REPLICATION": True, "EARLY_STOP_ENABLED": False,
        VAR_PLACEMENT: "clustered", VAR_ROUTING: "iterative", VAR_FANOUT: 2,
        VAR_CONNDROP: 0.0, VAR_ALPHA: 3, VAR_EF: 64, "ROUTING_DEBUG": False,
        "BIDIRECTIONAL_NEIGHBORS": True, "RANDOM_GOSSIP": False,
        "BOUNDED_VIEW": True, "PEER_SHUFFLE": True, "FARTHEST_ANCHOR": False,
        "FAR_RANDOM_BIAS": False, "INCREMENTAL_START": True, "LOOP_JITTER": True,
        "FAILURE_STRIKES": 3, "MEASURE_TOPOLOGY": False,
        "EXPLORE_FRACTION": 0.667, "MAX_NEIGHBORS_COEFF": 2.0,
        "RPC_TIMEOUT_BASE_S": RPC_TIMEOUT_BASE_S,
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


def run_condition(meta, cfg, regen):
    global _done
    _done += 1
    log(f"[{_done}/{_total}] {meta_str(meta)}  ({elapsed()})")
    set_config(**cfg)
    if regen and not generate_data():
        return False
    num_runs = meta.pop("num_runs", P["NUM_RUNS"])
    run_rows = []
    for r in range(num_runs):
        set_config(SEED=BASE_SEED + r)
        try:
            start_network(meta["num_nodes"])
            run_rows += run_evaluation()
        except Exception as e:
            log(f"  Fehler: {e!r}")
        finally:
            stop_network()
    rows = _aggregate(run_rows)
    for row in rows:
        row.update(meta)
        _all_rows.append(row)
    save()
    log(f"  -> {len(rows)} Zeilen, gesamt {len(_all_rows)}.")
    return True

ROUTING_EF_FIXED = 64

def build_final_overlay_plan():
    plan = []
    t1, t2 = P["TIER1"], P["TIER2"]

    def _nq(n):
        return P["NUM_QUERIES"] if n <= 200 else P["NUM_QUERIES_LARGE"]

    anchor = dict(
        ROUTING_STRATEGY="iterative", ROUTING_ALPHA=3, ROUTING_EF=ROUTING_EF_FIXED,
        BOUNDED_VIEW=True, PEER_SHUFFLE=True, BIDIRECTIONAL_NEIGHBORS=True,
        FARTHEST_ANCHOR=False, FAR_RANDOM_BIAS=False,
        INCREMENTAL_START=True, LOOP_JITTER=True, FAILURE_STRIKES=3,
        NUM_SEED_NODES=5, RANDOM_GOSSIP=False,
        EXPLORE_FRACTION=0.667, MAX_NEIGHBORS_COEFF=2.0,
        MEASURE_TOPOLOGY=True,
    )

    _old_rps = dict(BOUNDED_VIEW=False, PEER_SHUFFLE=False, RANDOM_GOSSIP=True,
                    INCREMENTAL_START=False, LOOP_JITTER=False, FAILURE_STRIKES=1,
                    NUM_SEED_NODES=1)

    def _derive_ttl(n, routing, ef, alpha):
        nav = math.ceil(math.log2(n)) + 2
        if routing == "iterative":
            return max(nav, math.ceil(ef / alpha)), f"iter{max(nav, math.ceil(ef / alpha))}"
        return nav, f"nav{nav}"

    def _cell(block, n, ds, overrides, runs, fmd=0, scen="none", needs_tox=False, **meta_kw):
        tot = _total_for(ds)
        routing = overrides.get("ROUTING_STRATEGY", "iterative")
        ef = overrides.get("ROUTING_EF", anchor["ROUTING_EF"])
        alpha = overrides.get("ROUTING_ALPHA", anchor["ROUTING_ALPHA"])
        ttl, ttl_mode = _derive_ttl(n, routing, ef, alpha)
        m = _meta(block, "clustered", routing, n, tot // n, fmd, True, scen, ds)
        m["num_runs"] = runs
        m["ttl_mode"] = ttl_mode
        m["routing_ef"] = ef
        m["rpc_timeout_s"] = overrides.get("RPC_TIMEOUT_BASE_S", RPC_TIMEOUT_BASE_S)
        m.update(meta_kw)
        c = base_cfg(n, [ttl], _nq(n), tot, DATASET=ds)
        c.update(anchor)
        c.update(overrides)
        return (m, c, needs_tox)

    _block_num     = {"routing": 1, "ablation": 2, "param": 3,
                      "seeds": 4, "scaling": 5, "robustheit": 6}
    _active_blocks = P.get("FINAL_ACTIVE_BLOCKS", [1, 2, 3, 4, 5, 6])
    _active_n      = P.get("FINAL_ACTIVE_N",      [200, 500, 750, 1000, 1250])

    def _add(cell):
        m = cell[0]
        if _block_num[m["block"]] in _active_blocks and m["num_nodes"] in _active_n:
            plan.append(cell)

    ROUTING_EF_BY_N = {50: 16, 200: 64}
    router_strategies = [
        ({}, t1),
        ({"ROUTING_STRATEGY": "greedy", "ROUTING_FANOUT": 3}, t1),
        ({"ROUTING_STRATEGY": "flood"}, 3),
        ({"ROUTING_STRATEGY": "random"}, 3),
    ]

    ablation_arms = [
        ("final",      {}),
        ("old_rps",    _old_rps),
        ("no_shuffle", {"PEER_SHUFFLE": False}),
        ("no_bidir",   {"BIDIRECTIONAL_NEIGHBORS": False}),
        ("unbounded",  {"BOUNDED_VIEW": False}),
        ("with_far",   {"FAR_RANDOM_BIAS": True}),
    ]

    for n, ef in ROUTING_EF_BY_N.items():
        for ds in ["ir"]:
            for ov, runs in router_strategies:
                overrides = dict(ov, ROUTING_EF=ef)
                _add(_cell("routing", n, ds, overrides, runs))

    for arm, ov in ablation_arms:
        _add(_cell("ablation", 500, "ir", ov, t1, arm=arm))

    for coeff in [1.0, 3.0]:
        _add(_cell("param", 500, "ir", {"MAX_NEIGHBORS_COEFF": coeff}, t2,
                          max_neighbors_coeff=coeff))
    for ef in [30, 48, 96, 128]:
        _add(_cell("param", 500, "ir", {"ROUTING_EF": ef}, t2))
    for alpha in [1, 6, 9]:
        _add(_cell("param", 500, "ir", {"ROUTING_ALPHA": alpha}, t2,
                          routing_alpha=alpha))
    for frac in [0.5, 0.75]:
        _add(_cell("param", 500, "ir", {"EXPLORE_FRACTION": frac}, t2,
                          explore_fraction=frac))

    for seeds in [1, 20]:
        _add(_cell("seeds", 500, "ir", {"NUM_SEED_NODES": seeds}, t2,
                          num_seed_nodes=seeds))

    _fault = dict(FAULT_INJECTION_ENABLED=True, FAULT_KILL_INTERVAL=8.0,
                  FAULT_KILL_PROBABILITY=0.4, FAULT_RESTART_DELAY=6.0)
    for fmd in [25, 50]:
        _add(_cell("robustheit", 500, "ir",
                          dict(_fault, FAULT_MAX_DOWN=fmd), t1, fmd=fmd))
    for scen in ["mid", "high"]:
        _add(_cell("robustheit", 500, "ir",
                          {"TOXIPROXY_ENABLED": True, "LATENCY_SCENARIO": scen}, t1,
                          scen=scen, needs_tox=True))
    for pct in [5.0, 10.0]:
        _add(_cell("robustheit", 500, "ir",
                          {"TOXIPROXY_ENABLED": True, "TOXIC_CONN_DROP_PCT": pct}, t1,
                          needs_tox=True, conn_drop_pct=pct))
    _add(_cell("robustheit", 500, "ir",
                      dict(_fault, FAULT_MAX_DOWN=25, TOXIPROXY_ENABLED=True, LATENCY_SCENARIO="mid"),
                      t1, fmd=25, scen="mid", needs_tox=True))

    for seeds in [1, 5, 20]:
        _add(_cell("seeds", 1000, "ir", {"NUM_SEED_NODES": seeds}, t2,
                          num_seed_nodes=seeds))

    _add(_cell("scaling", 1000, "ir", _old_rps, t2, arm="old_rps"))

    for seeds in [1, 5, 20]:
        _add(_cell("seeds", 1250, "ir", {"NUM_SEED_NODES": seeds}, t2,
                          num_seed_nodes=seeds))

    _add(_cell("scaling", 750, "ir", {}, t2, arm="final"))
    _add(_cell("scaling", 1250, "ir", {}, t2, arm="final"))

    return plan


def build_plan():
    return build_final_overlay_plan()

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

    plan = build_plan()
    _total = len(plan)
    if any(needs_tox for _, _, needs_tox in plan):
        start_toxiproxy()
    ns_in_plan = sorted({meta["num_nodes"] for meta, _, _ in plan})
    log(f"RUN={RUN}  Bloecke={_RUN_BLOCKS}  N={ns_in_plan}  "
        f"{_total} Bedingungen  CSV={os.path.basename(OUT_CSV)}  "
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