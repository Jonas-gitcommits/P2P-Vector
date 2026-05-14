import subprocess
import time
import sys
from config import NUM_NODES

START_PORT = 5000
processes = []

def start_network():
    print(f"Starte {NUM_NODES} P2P-Knoten...")
    for i in range(NUM_NODES):
        port = START_PORT + i
        
        # Der erste Knoten ist der Bootstrap-Knoten.
        # Alle anderen Knoten bekommen Port 5000 als Einstiegspunkt übergeben.
        bootstrap_port = str(START_PORT) if i > 0 else "None"
        
        # Befehl: python node.py <PORT> <BOOTSTRAP_PORT> <NODE_ID>
        cmd = [sys.executable, "node.py", str(port), bootstrap_port, str(i)]
        processes.append(subprocess.Popen(cmd))

    print("Netzwerk hochgefahren! Warte auf Initialisierung...")
    time.sleep(3)

def stop_network():
    print("Beende das P2P-Netzwerk...")
    for p in processes:
        p.terminate()

if __name__ == "__main__":
    try:
        start_network()
        print("Netzwerk läuft. (Drücke Ctrl+C zum Beenden)")
        
       # Einfach in einer Endlosschleife bleiben, bis der Benutzer mit Ctrl+C abbricht
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
        stop_network()