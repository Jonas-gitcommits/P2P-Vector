import subprocess
import time
import sys
import threading
import random

from config import (
    NUM_NODES,
    TOXIPROXY_ENABLED, REAL_PORT_START, PROXY_PORT_START,
    FAULT_INJECTION_ENABLED, FAULT_KILL_INTERVAL, FAULT_KILL_PROBABILITY,
    FAULT_MAX_DOWN, FAULT_RESTART_DELAY, FAULT_SEED,
    LATENCY_SCENARIO, SEED,
)

processes = []

_chaos_stop = threading.Event()
_chaos_thread = None


def _make_cmd(i: int) -> list:
    real_port = REAL_PORT_START + i
    proxy_port = PROXY_PORT_START + i if TOXIPROXY_ENABLED else real_port
    if i == 0:
        bootstrap_str = "None"
    else:
        bootstrap_str = str(PROXY_PORT_START if TOXIPROXY_ENABLED else REAL_PORT_START)
    return [sys.executable, "node.py", str(real_port), bootstrap_str, str(i), str(proxy_port), str(SEED + i)]


def start_network():
    print(f"Starte {NUM_NODES} P2P-Knoten...")
    for i in range(NUM_NODES):
        cmd = _make_cmd(i)
        processes.append([subprocess.Popen(cmd), cmd])

    print("Netzwerk hochgefahren! Warte auf Initialisierung...")
    time.sleep(3)

    if TOXIPROXY_ENABLED:
        from chaos.toxiproxy_setup import setup_proxies, apply_latency_scenario, add_connection_drops
        setup_proxies(NUM_NODES)
        apply_latency_scenario(NUM_NODES, LATENCY_SCENARIO)
        add_connection_drops(NUM_NODES)

    if FAULT_INJECTION_ENABLED:
        _start_chaos_loop()


def stop_network():
    print("Beende das P2P-Netzwerk...")
    _stop_chaos_loop()
    for proc, _ in processes:
        proc.terminate()
    if TOXIPROXY_ENABLED:
        from chaos.toxiproxy_setup import teardown_proxies
        teardown_proxies(NUM_NODES)


def _start_chaos_loop():
    global _chaos_thread
    _chaos_stop.clear()
    _chaos_thread = threading.Thread(target=_chaos_worker, daemon=True)
    _chaos_thread.start()


def _stop_chaos_loop():
    if _chaos_thread and _chaos_thread.is_alive():
        _chaos_stop.set()
        _chaos_thread.join(timeout=5)


def _chaos_worker():
    rng = random.Random(FAULT_SEED)
    
    down = {}

    while not _chaos_stop.is_set():
        _chaos_stop.wait(FAULT_KILL_INTERVAL)
        if _chaos_stop.is_set():
            break

        now = time.time()

        for idx in [k for k, (t, _) in down.items() if now - t >= FAULT_RESTART_DELAY]:
            _, cmd = down.pop(idx)
            new_proc = subprocess.Popen(cmd)
            processes[idx] = [new_proc, cmd]
            print(
                f"[{time.strftime('%H:%M:%S')}] [Chaos] Neustart Knoten {idx} "
                f"(Real-Port {REAL_PORT_START + idx})"
            )

        if len(down) < FAULT_MAX_DOWN and rng.random() < FAULT_KILL_PROBABILITY:
            candidates = [i for i in range(NUM_NODES) if i not in down]
            if candidates:
                idx = rng.choice(candidates)
                proc, cmd = processes[idx]
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                down[idx] = (time.time(), cmd)
                print(
                    f"[{time.strftime('%H:%M:%S')}] [Chaos] Abgeschaltet Knoten {idx} "
                    f"(Real-Port {REAL_PORT_START + idx})"
                )


if __name__ == "__main__":
    try:
        start_network()
        print("Netzwerk läuft. (Drücke Ctrl+C zum Beenden)")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
        stop_network()
