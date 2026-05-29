import json
import urllib.request
import urllib.error

from config import (
    TOXIPROXY_HOST, TOXIPROXY_API_PORT,
    REAL_PORT_START, PROXY_PORT_START,
    TOXIC_LATENCY_MS, TOXIC_LATENCY_JITTER_MS, TOXIC_CONN_DROP_PCT,
)

_BASE = f"http://{TOXIPROXY_HOST}:{TOXIPROXY_API_PORT}"

_START_HINT = (
    "Bitte starten mit:\n"
    "  Binary: toxiproxy-server -host 127.0.0.1 -port 8474"
)


def _api(method: str, path: str, body=None):
    url = _BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Toxiproxy nicht erreichbar ({TOXIPROXY_HOST}:{TOXIPROXY_API_PORT}): "
            f"{e.reason}\n{_START_HINT}"
        ) from None


def proxy_address(i: int) -> str:
    return f"{TOXIPROXY_HOST}:{PROXY_PORT_START + i}"


def setup_proxies(num_nodes: int):
    for i in range(num_nodes):
        name = f"node_{i}"
        try:
            _api("DELETE", f"/proxies/{name}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise RuntimeError(
                    f"Fehler beim Löschen von Proxy {name}: HTTP {e.code}"
                ) from e
        _api("POST", "/proxies", {
            "name": name,
            "listen": f"{TOXIPROXY_HOST}:{PROXY_PORT_START + i}",
            "upstream": f"{TOXIPROXY_HOST}:{REAL_PORT_START + i}",
            "enabled": True,
        })
        print(
            f"[Toxiproxy] Proxy {name}: "
            f":{PROXY_PORT_START + i} → :{REAL_PORT_START + i}"
        )


def add_latency_toxic(num_nodes: int):
    for i in range(num_nodes):
        name = f"node_{i}"
        _api("POST", f"/proxies/{name}/toxics", {
            "name": "latency_downstream",
            "type": "latency",
            "stream": "downstream",
            "toxicity": 1.0,
            "attributes": {
                "latency": TOXIC_LATENCY_MS,
                "jitter": TOXIC_LATENCY_JITTER_MS,
            },
        })
        print(
            f"[Toxiproxy] Latenz-Toxic auf {name}: "
            f"{TOXIC_LATENCY_MS}±{TOXIC_LATENCY_JITTER_MS} ms (downstream)"
        )


def apply_latency_scenario(num_nodes: int, scenario: str):
    """Entfernt vorhandene latency-toxics und setzt sie fürs Szenario neu.
    loss-toxics bleiben unberührt. Zur Laufzeit aufrufbar."""
    from config import LATENCY_PRESETS
    if scenario not in LATENCY_PRESETS:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")
    latency_ms, jitter_ms = LATENCY_PRESETS[scenario]
    for i in range(num_nodes):
        name = f"node_{i}"
        try:
            _api("DELETE", f"/proxies/{name}/toxics/latency_downstream")
        except (RuntimeError, urllib.error.HTTPError):
            pass
    if latency_ms == 0 and jitter_ms == 0:
        print(f"[Toxiproxy] Szenario '{scenario}': keine Latenz-Toxics")
        return
    for i in range(num_nodes):
        name = f"node_{i}"
        _api("POST", f"/proxies/{name}/toxics", {
            "name": "latency_downstream", "type": "latency",
            "stream": "downstream", "toxicity": 1.0,
            "attributes": {"latency": latency_ms, "jitter": jitter_ms},
        })
    print(f"[Toxiproxy] '{scenario}': {latency_ms}±{jitter_ms} ms auf {num_nodes} Proxies")


def add_connection_drops(num_nodes: int):
    if TOXIC_CONN_DROP_PCT <= 0:
        return
    toxicity = TOXIC_CONN_DROP_PCT / 100.0
    for i in range(num_nodes):
        name = f"node_{i}"
        _api("POST", f"/proxies/{name}/toxics", {
            "name": "loss_downstream",
            "type": "limit_data",
            "stream": "downstream",
            "toxicity": toxicity,
            "attributes": {"bytes": 1},
        })
        print(
            f"[Toxiproxy] Verbindungsabbruch-Toxic auf {name}: "
            f"{TOXIC_CONN_DROP_PCT}% Verbindungsabbrüche (limit_data, 1 Byte, downstream)"
        )


def remove_all_toxics(num_nodes: int):
    for i in range(num_nodes):
        name = f"node_{i}"
        try:
            toxics = _api("GET", f"/proxies/{name}/toxics")
            for toxic in toxics:
                try:
                    _api("DELETE", f"/proxies/{name}/toxics/{toxic['name']}")
                except (RuntimeError, urllib.error.HTTPError):
                    pass
        except (RuntimeError, urllib.error.HTTPError):
            pass


def teardown_proxies(num_nodes: int):
    remove_all_toxics(num_nodes)
    for i in range(num_nodes):
        name = f"node_{i}"
        try:
            _api("DELETE", f"/proxies/{name}")
        except (RuntimeError, urllib.error.HTTPError):
            pass
    print(f"[Toxiproxy] {num_nodes} Proxies entfernt.")
