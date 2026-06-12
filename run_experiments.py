import os, re, sys, csv, time, signal, subprocess, urllib.request
from datetime import datetime
import numpy as np
from scipy.stats import t as _t_dist
from evaluate import run_evaluation

HERE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable

PROFILE    = "standard"
SMOKE_TEST = False

IR_CORPUS_SIZE = 200_000
BASE_SEED = 1234

EARLY_STOP_THRESHOLDS = {"sift": 65000.0, "ir": 0.90}
FAULT_PROFILE = dict(FAULT_KILL_INTERVAL=8.0, FAULT_KILL_PROBABILITY=0.4, FAULT_RESTART_DELAY=6.0)

PROFILES = {
    "standard": dict(
        DATASETS=["ir", "sift"],
        TOTAL_VECTORS=20000, N_BASE=20, N_LIST_SCALE=[10, 20, 30, 50],
        NUM_QUERIES=300, NUM_QUERIES_LATENCY=150, NUM_RUNS=5,
        GOSSIP_WARMUP_S=30,
        TTL_CORE=[2, 4, 6], TTL_CHURN=[4, 6], TTL_LATENCY=[4],
        FANOUT_LIST=[1, 2, 3, 4], WARMUP_LIST=[0, 10, 30],
        CONNDROP_LIST=[0, 10, 30], FAULT_LEVELS=[0, 2, 4],
        NETWORK_BOOT_WAIT=6,
    ),
    "long": dict(
        DATASETS=["ir", "sift"],
        TOTAL_VECTORS=200000, N_BASE=20, N_LIST_SCALE=[10, 20, 50, 100, 200],
        NUM_QUERIES=1000, NUM_QUERIES_LATENCY=400, NUM_RUNS=5,
        GOSSIP_WARMUP_S=40,
        TTL_CORE=[2, 4, 6, 8], TTL_CHURN=[4, 6, 8], TTL_LATENCY=[4, 6],
        FANOUT_LIST=[1, 2, 3, 4, 6, 8], WARMUP_LIST=[0, 5, 10, 20, 40, 60],
        CONNDROP_LIST=[0, 5, 10, 20, 40], FAULT_LEVELS=[0, 2, 4, 8],
        NETWORK_BOOT_WAIT=12,
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
    extra = "".join(f" {k}={m[k]}" for k in ("fanout", "gossip_warmup_s", "conn_drop_pct", "early_stop") if k in m)
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
    time.sleep(P["NETWORK_BOOT_WAIT"] + 0.05 * n)

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
        })
    return result


def run_condition(meta, cfg, regen):
    global _done
    _done += 1
    log(f"[{_done}/{_total}] {meta_str(meta)}  ({elapsed()})")
    set_config(**cfg)
    if regen and not generate_data():
        return False
    run_rows = []
    for r in range(P["NUM_RUNS"]):
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

    for ds in datasets:
        total = _total_for(ds)
        nb, vb = P["N_BASE"], total // P["N_BASE"]

        for placement in ["clustered", "contiguous"]:
            for routing in ["greedy", "random", "flood", "iterative"]:
                m = _meta("ablation", placement, routing, nb, vb, 0, True, "none", ds)
                c = base_cfg(nb, P["TTL_CORE"], P["NUM_QUERIES"], total, DATASET=ds,
                             **{VAR_PLACEMENT: placement, VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, False))

        for n in P["N_LIST_SCALE"]:
            for routing in ["greedy", "flood", "iterative"]:
                m = _meta("scale", "clustered", routing, n, total // n, 0, True, "none", ds)
                c = base_cfg(n, P["TTL_CORE"], P["NUM_QUERIES"], total, DATASET=ds,
                             **{VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, False))

        fmax = max(P["FAULT_LEVELS"]) or 4
        for routing in ["greedy", "iterative"]:
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

        for scen in ["none", "mid", "high"]:
            for routing in ["greedy", "flood", "iterative"]:
                m = _meta("latency", "clustered", routing, nb, vb, 0, True, scen, ds)
                c = base_cfg(nb, P["TTL_LATENCY"], P["NUM_QUERIES_LATENCY"], total,
                             DATASET=ds, TOXIPROXY_ENABLED=True, LATENCY_SCENARIO=scen,
                             **{VAR_ROUTING: routing}, **_iter(routing))
                plan.append((m, c, True))

        for fan in P["FANOUT_LIST"]:
            m = _meta("fanout", "clustered", "greedy", nb, vb, 0, True, "none", ds)
            m["fanout"] = fan
            c = base_cfg(nb, P["TTL_CORE"], P["NUM_QUERIES"], total,
                         DATASET=ds, **{VAR_FANOUT: fan})
            plan.append((m, c, False))

        for w in P["WARMUP_LIST"]:
            m = _meta("gossip", "clustered", "greedy", nb, vb, 0, True, "none", ds)
            m["gossip_warmup_s"] = w
            c = base_cfg(nb, [4], P["NUM_QUERIES"], total, DATASET=ds, **{VAR_WARMUP: w})
            plan.append((m, c, False))

        for pct in P["CONNDROP_LIST"]:
            m = _meta("conndrop", "clustered", "greedy", nb, vb, 0, True, "none", ds)
            m["conn_drop_pct"] = pct
            c = base_cfg(nb, [4], P["NUM_QUERIES_LATENCY"], total,
                         DATASET=ds, TOXIPROXY_ENABLED=True, **{VAR_CONNDROP: float(pct)})
            plan.append((m, c, True))

        ttl_h = P["TTL_CORE"][len(P["TTL_CORE"]) // 2:]
        for routing in ["greedy", "iterative"]:
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
