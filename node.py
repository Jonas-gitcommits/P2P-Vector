import asyncio
import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import faiss
import random
import sys

MAX_NEIGHBORS = 8

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
    
    def add_neighbor_edge(self, ip, port, vector):
        target = f"{ip}:{port}"
        if target not in self.neighbors:
            self.neighbors[target] = []
        self.neighbors[target].append(vector)
 
        if len(self.neighbors) > MAX_NEIGHBORS:
            neighbor_distances = []
            for n_target, n_vectors in self.neighbors.items():
                vec_np = np.array([n_vectors[-1]], dtype=np.float32)
                if self.local_index.ntotal > 0:
                    dists, _ = self.local_index.search(vec_np, 1)
                    dist = float(dists[0][0])
                else:
                    dist = float("inf")
                neighbor_distances.append((dist, n_target, n_vectors))
 
            neighbor_distances.sort(key=lambda x: x[0])
            best_neighbors = neighbor_distances[:6]
            remaining = neighbor_distances[6:]
            random_picks = random.sample(remaining, 2)
 
            self.neighbors = {}
            for _, t, v_list in best_neighbors + random_picks:
                self.neighbors[t] = v_list


class VectorStoreServicer(p2p_pb2_grpc.VectorStoreServicer):
    def __init__(self, port, local_graph, router):
        self.port = port
        self.local_graph = local_graph
        self.router = router

    async def SearchSimilar(self, request, context):
        query_vec = np.frombuffer(request.query.values, dtype=np.float32).tolist()
        visited = list(request.visited_peers)
        
        if request.sender_port > 0 and request.sender_port != 9999:
            self.local_graph.add_neighbor_edge(request.sender_ip, request.sender_port, query_vec)
           
        local_res = self.local_graph.search_local(query_vec, request.k, "127.0.0.1", self.port)
        combined_res = list(local_res)

        if request.ttl > 0:
            remote_res = await self.router.distributed_search(
                self.local_graph.neighbors,
                query_vec,
                request.k,
                request.ttl,
                visited
            )
            combined_res.extend(remote_res)

        combined_res.sort(key=lambda x: x[2])
        final_best_k = combined_res[:request.k]

        response = p2p_pb2.SearchResponse()
        for ip, port, dist in final_best_k:
            p = response.nearest_peers.add()
            p.ip = ip
            p.port = port
            response.distances.append(dist)
            
        return response
    
async def Ping(self, request, context):
        return p2p_pb2.PingResponse(
            alive=True, 
            neighbor_count=len(self.local_graph.neighbors)
        )

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

    if bootstrap_port and bootstrap_port != "None":
        local_graph.add_neighbor_edge("127.0.0.1", int(bootstrap_port), [0.0] * 128)

    from protocol import DistributedRouter
    router = DistributedRouter("127.0.0.1", port)

    asyncio.create_task(router.health_check_loop(local_graph))
    
    server = grpc.aio.server()
    
    p2p_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(port, local_graph, router), server
    )
    server.add_insecure_port(f'[::]:{port}')
    
    print(f"[Node {port}] Online.")
    
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    b = sys.argv[2] if len(sys.argv) > 2 else None
    n_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    asyncio.run(serve(p, b, n_id))