set -euo pipefail

N="${1:?Usage: $0 <anzahl_shards>}"
REDIS_BIN="${REDIS_BIN:-redis-server}"
CLI_BIN="${CLI_BIN:-redis-cli}"
BASE_PORT="${BASE_PORT:-20000}"
CLUSTER_PORT_BASE="${CLUSTER_PORT_BASE:-$((BASE_PORT + 10000))}"
CLUSTER_NODE_TIMEOUT_MS="${CLUSTER_NODE_TIMEOUT_MS:-30000}"
REDIS_MODULES_DIR="${REDIS_MODULES_DIR:-/usr/lib/redis/modules}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNDIR="${WORKDIR}/run"
LOGDIR="${WORKDIR}/logs"
PIDFILE="${RUNDIR}/pids.txt"

mkdir -p "${RUNDIR}" "${LOGDIR}"
: > "${PIDFILE}"

echo "[setup] Pruefe Redis-Version ..."
VERSION_LINE="$("${REDIS_BIN}" --version)"
echo "  ${VERSION_LINE}"
VERSION="$(echo "${VERSION_LINE}" | grep -oP 'v=\K[0-9]+' || echo 0)"
if [[ "${VERSION}" -lt 8 ]]; then
  echo "[FEHLER] Redis-Version < 8. Cluster-weite FT.SEARCH-Koordination ist erst ab 8.0 im Kern enthalten."
  exit 1
fi

if [[ -f "${REDIS_MODULES_DIR}/redisearch.so" ]]; then
  echo "[setup] Search-Modul gefunden: ${REDIS_MODULES_DIR}/redisearch.so (wird auf jedem Knoten geladen)"
else
  echo "[WARNUNG] Kein redisearch.so unter ${REDIS_MODULES_DIR} gefunden."
  echo "          FT.CREATE/FT.SEARCH werden dann fehlschlagen. Pruefen mit:"
  echo "            ls /usr/lib/redis/modules/"
  echo "          und ggf. REDIS_MODULES_DIR=<pfad> setzen."
fi

echo "[setup] Starte ${N} native Redis-Instanzen ab Port ${BASE_PORT} ..."

for i in $(seq 0 $((N - 1))); do
  PORT=$((BASE_PORT + i))
  BUS_PORT=$((CLUSTER_PORT_BASE + i))
  NODEDIR="${RUNDIR}/node-${PORT}"
  mkdir -p "${NODEDIR}"

  CONF="${NODEDIR}/redis.conf"
  cat > "${CONF}" <<EOF
port ${PORT}
cluster-enabled yes
cluster-port ${BUS_PORT}
cluster-config-file ${NODEDIR}/nodes.conf
cluster-node-timeout ${CLUSTER_NODE_TIMEOUT_MS}
appendonly no
save ""
dir ${NODEDIR}
daemonize no
pidfile ${NODEDIR}/redis.pid
logfile ${LOGDIR}/node-${PORT}.log
bind 127.0.0.1
protected-mode no
EOF

  if [[ -f "${REDIS_MODULES_DIR}/redisearch.so" ]]; then
    echo "loadmodule ${REDIS_MODULES_DIR}/redisearch.so WORKERS 0" >> "${CONF}"
  fi

  "${REDIS_BIN}" "${CONF}" > /dev/null 2>&1 &
  PID=$!
  echo "${PID}" >> "${PIDFILE}"
  echo "  Node ${i}: Port ${PORT} (Bus ${BUS_PORT}, PID ${PID})"
done

echo "[setup] Warte auf Instanzen ..."
sleep 2

for i in $(seq 0 $((N - 1))); do
  PORT=$((BASE_PORT + i))
  if ! "${CLI_BIN}" -p "${PORT}" ping > /dev/null 2>&1; then
    echo "[FEHLER] Node auf Port ${PORT} antwortet nicht. Log: ${LOGDIR}/node-${PORT}.log"
    exit 1
  fi
done

echo "[setup] Alle ${N} Instanzen erreichbar. Forme Cluster ..."

NODE_LIST=""
for i in $(seq 0 $((N - 1))); do
  PORT=$((BASE_PORT + i))
  NODE_LIST="${NODE_LIST} 127.0.0.1:${PORT}"
done

"${CLI_BIN}" --cluster create ${NODE_LIST} --cluster-replicas 0 --cluster-yes > "${LOGDIR}/cluster-create.log" 2>&1 || {
  echo "[FEHLER] Cluster-Erstellung fehlgeschlagen, siehe ${LOGDIR}/cluster-create.log"
  exit 1
}

sleep 1

ASSIGNED="$("${CLI_BIN}" -p "${BASE_PORT}" cluster info | grep cluster_slots_assigned | tr -d '\r' | cut -d: -f2)"

if [[ "${ASSIGNED}" == "16384" ]]; then
  echo "[OK] All 16384 slots covered"
else
  echo "[FEHLER] Nur ${ASSIGNED}/16384 Slots zugewiesen. Siehe ${LOGDIR}/cluster-create.log"
  exit 1
fi

echo "[setup] Warte auf Gossip-Konvergenz (cluster_state:ok auf allen Knoten) ..."
MAX_TRIES=$(( 20 + N ))
for attempt in $(seq 1 "${MAX_TRIES}"); do
  ALL_OK=1
  for i in $(seq 0 $((N - 1))); do
    PORT=$((BASE_PORT + i))
    STATE=$("${CLI_BIN}" -p "${PORT}" cluster info | grep cluster_state | tr -d '\r' | cut -d: -f2)
    if [[ "${STATE}" != "ok" ]]; then
      ALL_OK=0
      break
    fi
  done
  if [[ "${ALL_OK}" -eq 1 ]]; then
    echo "[OK] cluster_state:ok auf allen ${N} Knoten (nach ${attempt}s)"
    break
  fi
  if [[ "${attempt}" -eq "${MAX_TRIES}" ]]; then
    echo "[FEHLER] cluster_state nach ${MAX_TRIES}s auf mindestens einem Knoten nicht 'ok'."
    echo "         Manuell pruefen: redis-cli -p ${BASE_PORT} cluster info"
    exit 1
  fi
  sleep 1
done

echo "${BASE_PORT}" > "${RUNDIR}/base_port.txt"
echo "${N}" > "${RUNDIR}/n_shards.txt"
echo "[setup] Cluster bereit. Erster Knoten: 127.0.0.1:${BASE_PORT}"