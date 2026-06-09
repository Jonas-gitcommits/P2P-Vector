# P2P Vector Store

Verteilter Vektorspeicher mit HNSW-Graphen und Gossip-Routing.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Datensatz vorbereiten

In `config.py` den Datensatz wählen (`DATASET = 'sift'` oder `'ir'`).

**SIFT** (128-dim):
```bash
python download_sift.py   # einmalig
python generate_data.py
```

**MS MARCO / IR** (384-dim):
```bash
python ir_dataset.py      # einmalig
python generate_data.py
```

`generate_data.py` schreibt `dataset.npy`, `queries.npy` und `partition.npy`.

## Ausführen

### Ohne Netzwerksimulation (TOXIPROXY_ENABLED = False)

Terminal 1 – Netzwerk starten:
```bash
python simulator.py
```

Terminal 2 – Evaluation:
```bash
python evaluate.py
```

### Mit Toxiproxy (TOXIPROXY_ENABLED = True)

Toxiproxy muss vor dem Simulator laufen:

```bash
# Terminal 1 – Toxiproxy-Server
./toxiproxy_2.12.0_linux_amd64

# Terminal 2 – Netzwerk
python simulator.py

# Terminal 3 – Evaluation
python evaluate.py
```

Latenz und Verbindungsabbrüche werden über `LATENCY_SCENARIO` und
`TOXIC_CONN_DROP_PCT` in `config.py` konfiguriert.

Standard-Sicherheitslimit bei Linux ist bei 1024 gleichzeitig offenen Sockets. Wenn dieses Limit erreicht wird, stürzt gRPC ab.
Limit hochsetzen mit: 
```bash
ulimit -n 8192
```