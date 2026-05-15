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
        self.neighbors = {}

    def insert_local(self, vector):
        vec_np = np.array([vector], dtype=np.float32)
        self.local_index.add(vec_np)

    def search_local(self, query_vector, k, my_ip, my_port):
        """Führt eine lokale FAISS-Suche durch."""
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
    def __init__(self, port, local_graph, router, neighbor_port):
        self.port = port
        self.local_graph = local_graph
        self.router = router
        self.neighbor_port = neighbor_port

    async def SearchSimilar(self, request, context):
        
        query_vec = np.frombuffer(request.query.values, dtype=np.float32).tolist()

        local_res = self.local_graph.search_local(query_vec, request.k, "127.0.0.1", self.port)
        
        network_res = []
        if self.neighbor_port:
            network_res = await self.router.ask_neighbor(self.neighbor_port, query_vec, request.k)
        
        all_results = local_res + network_res
        all_results.sort(key=lambda x: x[2])
        
        unique_results = []
        seen = set()
        for ip, port, dist in all_results:
            key = (ip, port, round(dist, 5))
            if key not in seen:
                seen.add(key)
                unique_results.append((ip, port, dist))
                
        best_k = unique_results[:request.k]

        response = p2p_pb2.SearchResponse()
        for ip, port, dist in best_k:
            p = response.nearest_peers.add()
            p.ip = ip
            p.port = port
            response.distances.append(dist)
            
        return response

async def serve(port, bootstrap_port=None, node_id=0):
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
       
    from protocol import DistributedRouter
    router = DistributedRouter()
    
    server = grpc.aio.server()
    p2p_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(port, local_graph, router, bootstrap_port), server
    )
    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    b = sys.argv[2] if len(sys.argv) > 2 else None
    n_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    asyncio.run(serve(p, b, n_id))