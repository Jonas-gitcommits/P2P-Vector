import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import time
import random
import faiss
from config import NUM_NODES, SUBSET_SIZE

NUM_QUERIES = 100
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

    print("Warte 5s für Gossip-Konvergenz...")
    time.sleep(5)
 
    random.seed(42)
    entry_nodes = [random.choice(ALL_NODES) for _ in queries]
 
    channels = {n: grpc.insecure_channel(n) for n in ALL_NODES}
    stubs = {n: p2p_pb2_grpc.VectorStoreStub(channels[n]) for n in ALL_NODES}
 
    latencies = []
    recalls = []
    errors = 0
 
    print(f"Starte {NUM_QUERIES} Queries (TTL={TTL}, K={K})...")
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
        except grpc.RpcError as e:
            errors += 1
           
    for ch in channels.values():
        ch.close()
 
    print("\n" + "=" * 60)
    print(f"Queries erfolgreich: {len(latencies)}/{NUM_QUERIES}  (Fehler: {errors})")
    if latencies:
        print(f"Recall:      {np.mean(recalls) * 100:6.2f} %")
        print(f"Avg Latenz:  {np.mean(latencies):6.2f} ms")
        print(f"P95 Latenz:  {np.percentile(latencies, 95):6.2f} ms")
    else:
        print("Keine erfolgreichen Queries.")
    print("=" * 60)
 
 
if __name__ == "__main__":
    run_evaluation()
 
