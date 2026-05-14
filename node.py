import asyncio
import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import faiss
import sys

class LocalGraphState:
    def __init__(self, dimension=128, M=32):
        self.dimension = dimension
        self.local_index = faiss.IndexHNSWFlat(dimension, M)

    def insert_local(self, vector):
        vec_np = np.array([vector], dtype=np.float32)
        self.local_index.add(vec_np)

    def search_local(self, query_vector, k, my_ip, my_port):
        """Führt eine simple lokale FAISS-Suche durch."""
        query_np = np.array([query_vector], dtype=np.float32)
        results = []

        if self.local_index.ntotal > 0:
            search_k = min(k, self.local_index.ntotal)
            dist, _ = self.local_index.search(query_np, search_k)
            for d in dist[0]:
                results.append((my_ip, my_port, float(d)))

        results.sort(key=lambda x: x[2])
        return results[:k]


class VectorStoreServicer(p2p_pb2_grpc.VectorStoreServicer):
    def __init__(self, port, local_graph):
        self.port = port
        self.local_graph = local_graph

    async def SearchSimilar(self, request, context):
        # 1. Anfragevektor extrahieren
        query_vec = np.frombuffer(request.query.values, dtype=np.float32).tolist()
        
        # 2. Lokale Suche durchführen
        local_res = self.local_graph.search_local(query_vec, request.k, "127.0.0.1", self.port)
        
        # 3. Antwort mit den nächsten Peers erstellen
        response = p2p_pb2.SearchResponse()
        for ip, port, dist in local_res:
            p = response.nearest_peers.add()
            p.ip = ip
            p.port = port
            response.distances.append(dist)
            
        return response


async def serve(port, node_id=0):
    local_graph = LocalGraphState()

    try:
        from config import VECTORS_PER_NODE
        dataset = np.load("dataset.npy")
        chunk_size = VECTORS_PER_NODE
        
        start_idx = node_id * chunk_size
        my_chunk = dataset[start_idx:start_idx + chunk_size]
        for vec in my_chunk:
            local_graph.insert_local(vec.tolist())

        print(f"[Node {port}] ID {node_id}: {len(my_chunk)} Vektoren geladen.")
    except FileNotFoundError:
        print(f"[Node {port}] Fehler: dataset.npy nicht gefunden!")

    # gRPC Server starten
    server = grpc.aio.server()
    p2p_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(port, local_graph), server
    )
    server.add_insecure_port(f'[::]:{port}')
    print(f"Starte gRPC Server auf Port {port}...")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    # Argumente: Port und Node-ID (bestimmt den Daten-Chunk)
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    asyncio.run(serve(p, n_id))