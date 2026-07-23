set -uo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNDIR="${WORKDIR}/run"
PIDFILE="${RUNDIR}/pids.txt"

if [[ ! -f "${PIDFILE}" ]]; then
  echo "[teardown] Keine PID-Datei gefunden (${PIDFILE})."
  exit 0
fi

echo "[teardown] Beende Instanzen ueber PID-Datei ..."

while read -r PID; do
  [[ -z "${PID}" ]] && continue
  if kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}"
    echo "  PID ${PID}: SIGTERM gesendet"
  else
    echo "  PID ${PID}: lief nicht mehr"
  fi
done < "${PIDFILE}"

sleep 1

while read -r PID; do
  [[ -z "${PID}" ]] && continue
  if kill -0 "${PID}" 2>/dev/null; then
    kill -9 "${PID}" 2>/dev/null
    echo "  PID ${PID}: hart beendet (SIGKILL)"
  fi
done < "${PIDFILE}"

rm -rf "${RUNDIR}"
echo "[teardown] Aufgeraeumt: ${RUNDIR} entfernt."
