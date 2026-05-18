import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import time

def test_search():
    print("Lade eine Query...")
    queries = np.load("queries.npy")
    query_vec = queries[0] 
    query_bytes = np.array(query_vec, dtype=np.float32).tobytes()

    target = "127.0.0.1:5001"
    print(f"Verbinde mit {target}...\n")

    channel = grpc.insecure_channel(target)
    stub = p2p_pb2_grpc.VectorStoreStub(channel)

    print("--- Isoliert: Lokale Suche (TTL=0) ---")
    req_local = p2p_pb2.SearchRequest(
        query=p2p_pb2.Vector(values=query_bytes),
        k=5,
        ttl=0,
        visited_peers=[]
    )
    start = time.time()
    res_local = stub.SearchSimilar(req_local)
    print(f"Dauer: {(time.time() - start)*1000:.2f} ms")
    print("Beste Ergebnisse (Peer -> Distanz):")
    for p, d in zip(res_local.nearest_peers, res_local.distances):
        print(f"  {p.ip}:{p.port} -> Distanz: {d:.2f}")

    print("\n--- Netzwerk: Verteilte Suche (TTL=2) ---")
    req_dist = p2p_pb2.SearchRequest(
        query=p2p_pb2.Vector(values=query_bytes),
        k=5,
        ttl=2,
        visited_peers=[]
    )
    start = time.time()
    res_dist = stub.SearchSimilar(req_dist)
    print(f"Dauer: {(time.time() - start)*1000:.2f} ms")
    print("Beste Ergebnisse (Peer -> Distanz):")
    for p, d in zip(res_dist.nearest_peers, res_dist.distances):
        print(f"  {p.ip}:{p.port} -> Distanz: {d:.2f}")

if __name__ == '__main__':
    test_search()