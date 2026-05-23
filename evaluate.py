import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import time
import random
import faiss
from config import NUM_NODES, SUBSET_SIZE

NUM_QUERIES = 100
NUM_RUNS = 3
DIMENSION = 128
K = 3
TTL = 4
START_PORT = 5000
ALL_NODES = [f"127.0.0.1:{START_PORT + i}" for i in range(NUM_NODES)]
 
def build_ground_truth_max_dists(dataset, queries, full_gt):
    """Berechnet für jede Query die maximale Distanz zu den Top-K Ground-Truth-Vektoren im Subset."""
    central_index = None
    gt_max_dists = []
 
    for qi, q in enumerate(queries):

        gt_indices_in_subset = [idx for idx in full_gt[qi] if idx < SUBSET_SIZE]
 
        if len(gt_indices_in_subset) >= K:
            top_k_vecs = dataset[gt_indices_in_subset[:K]]
            
            dists = np.linalg.norm(top_k_vecs - q, axis=1) ** 2
            gt_max_dists.append(float(dists.max()))
        else:
           
            if central_index is None:
                central_index = faiss.IndexFlatL2(DIMENSION)
                central_index.add(dataset)
            d, _ = central_index.search(
                np.array([q], dtype=np.float32), K
            )
            gt_max_dists.append(float(d[0][-1]))
 
    return gt_max_dists
 
def get_alive_nodes(nodes):
    alive = []
    for n in nodes:
        try:
            channel = grpc.insecure_channel(n)
            stub = p2p_pb2_grpc.VectorStoreStub(channel)
            stub.Ping(p2p_pb2.PingRequest(), timeout=1.0)
            alive.append(n)
        except grpc.RpcError:
            pass
        finally:
            channel.close()
    return alive
 
def run_evaluation():
    print("Lade Datensatz...")
    dataset = np.load("dataset.npy").astype(np.float32)
    all_queries = np.load("queries.npy").astype(np.float32)
    full_gt = np.load("ground_truth.npy")
    queries = all_queries[:NUM_QUERIES]
    print(f"  dataset: {dataset.shape}, queries: {queries.shape}")
    print(f"  SUBSET_SIZE = {SUBSET_SIZE} (NUM_NODES={NUM_NODES})")
 
    print("Berechne Ground-Truth-Distanzen...")
    gt_max_dists = build_ground_truth_max_dists(dataset, queries, full_gt)
    print(f"  Fertig: {len(gt_max_dists)} Distanzen.\n")

    print("Warte 10s für Gossip-Konvergenz...")
    time.sleep(10)

    alive_nodes = get_alive_nodes(ALL_NODES)
    print(f"Erreichbar: {len(alive_nodes)}/{len(ALL_NODES)}")
    
    if not alive_nodes:
        print("Keine Knoten erreichbar! Abbruch.")
        return

    channels = {n: grpc.insecure_channel(n) for n in alive_nodes}
    stubs = {n: p2p_pb2_grpc.VectorStoreStub(channels[n]) for n in alive_nodes}

    run_recalls = []
    run_latencies = []
    all_latencies = []

    for run_id in range(NUM_RUNS):
        random.seed(42 + run_id)
        
        entry_nodes = [random.choice(alive_nodes) for _ in queries]

        latencies = []
        recalls = []

        print(f"Run {run_id + 1}/{NUM_RUNS} ({NUM_QUERIES} Queries, TTL={TTL}, K={K})...")
        for i, query_vector in enumerate(queries):
            entry_node = entry_nodes[i]
            gt_max = gt_max_dists[i]

            query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
            vec = p2p_pb2.Vector(values=query_bytes)
            request = p2p_pb2.SearchRequest(
                query=vec,
                k=K,
                ttl=TTL,
                visited_peers=[],
                sender_ip="127.0.0.1",
                sender_port=9999,
            )

            try:
                t0 = time.time()
                response = stubs[entry_node].SearchSimilar(request, timeout=5.0)
                latencies.append((time.time() - t0) * 1000)
                matches = sum(1 for d in response.distances if d <= gt_max + 1e-5)
                recalls.append(min(matches, K) / K)
            except grpc.RpcError:
                pass

        if recalls:
            run_recalls.append(np.mean(recalls) * 100)
            run_latencies.append(np.mean(latencies))
            all_latencies.extend(latencies)

    for ch in channels.values():
        ch.close()

    if run_recalls:
        print("\n" + "=" * 60)
        print(f"Recall:      {np.mean(run_recalls):6.2f} %  "
              f"(Min: {np.min(run_recalls):.2f} %, Max: {np.max(run_recalls):.2f} %)")
        print(f"Avg Latenz:  {np.mean(all_latencies):6.2f} ms")
        print(f"P95 Latenz:  {np.percentile(all_latencies, 95):6.2f} ms")
        print("=" * 60)
 
 
if __name__ == "__main__":
    run_evaluation()
 
