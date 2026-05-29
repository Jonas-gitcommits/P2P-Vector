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
            dist, idx = self.local_index.search(query_np, search_k)
            for d, i in zip(dist[0], idx[0]):
                if i >= 0 and np.isfinite(d):
                    results.append((my_ip, my_port, float(d)))

        results.sort(key=lambda x: x[2])
        return results[:k]
    
    def get_my_latest_vector(self):
        if self.local_index.ntotal == 0:
            return [0.0] * self.dimension
        idx = random.randint(0, self.local_index.ntotal - 1)
        return self.local_index.reconstruct(idx).tolist()

    def evaluate_next_hop(self, query_vector, visited_peers, best_dist_so_far=None, fanout=2):
        query_np = np.array(query_vector, dtype=np.float32)
        valid_neighbors = []
        for target, vectors in self.neighbors.items():
            if target in visited_peers:
                continue
            vec_np = np.array(vectors[-1], dtype=np.float32)
            dist = float(np.sum((query_np - vec_np) ** 2))
            valid_neighbors.append((target, dist))

        if not valid_neighbors:
            return {"action": "stop", "targets": []}

        valid_neighbors.sort(key=lambda x: x[1])
        best_targets = [n[0] for n in valid_neighbors[:fanout]]
        return {"action": "hop", "targets": best_targets}

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
            random_picks = random.sample(remaining, min(2, len(remaining)))
 
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
        
        if (request.sender_port > 0 and request.sender_port != 9999
                and not (request.sender_ip == "127.0.0.1" and request.sender_port == self.port)):
            self.local_graph.add_neighbor_edge(request.sender_ip, request.sender_port, query_vec)
           
        is_entry = (request.sender_port == 9999)
        fanout_k = request.fanout_k if request.fanout_k > 0 else max(request.k * 4, 20)
        local_budget = max(fanout_k, request.k)

        local_res = self.local_graph.search_local(query_vec, local_budget, "127.0.0.1", self.port)
        combined_res = list(local_res)

        kth_dist = float(local_res[-1][2]) if local_res else request.kth_dist
        my_id = f"127.0.0.1:{self.port}"
        rpc_count = 1
        visited_nodes = {my_id}

        if request.ttl > 0:
            remote_result = await self.router.distributed_search(
                self.local_graph,
                query_vec,
                request.k,
                request.ttl,
                visited,
                kth_dist=kth_dist,
                fanout_k=fanout_k,
                early_stop_threshold=request.early_stop_threshold,
            )
            combined_res.extend(remote_result["peers"])
            rpc_count += remote_result["rpc_count"]
            visited_nodes |= remote_result["visited_nodes"]

        combined_res.sort(key=lambda x: x[2])

        seen = set()
        deduped = []
        for ip, port, dist in combined_res:
            dkey = round(dist, 5)
            if dkey in seen:
                continue
            seen.add(dkey)
            deduped.append((ip, port, dist))

        final = deduped[:request.k] if is_entry else deduped[:max(fanout_k, request.k)]

        response = p2p_pb2.SearchResponse()
        for ip, port, dist in final:
            p = response.nearest_peers.add()
            p.ip = ip
            p.port = port
            response.distances.append(dist)
        response.rpc_count = rpc_count
        response.visited_nodes.extend(sorted(visited_nodes))

        return response
    
    async def Ping(self, request, context):
        return p2p_pb2.PingResponse(
            alive=True, 
            neighbor_count=len(self.local_graph.neighbors)
        )

async def serve(real_port, bootstrap_port=None, node_id=0, proxy_port=None):
    proxy_port = proxy_port or real_port
    local_graph = LocalGraphState()

    try:
        from config import VECTORS_PER_NODE, NUM_NODES, REPLICATION
        dataset = np.load("dataset.npy")
        chunk_size = VECTORS_PER_NODE

        start_idx = node_id * chunk_size
        if start_idx + chunk_size > len(dataset):
            raise RuntimeError(
                f"dataset.npy zu klein!"
                f"Bitte `python generate_data.py` erneut ausführen."
            )
        my_chunk = dataset[start_idx:start_idx + chunk_size]
        for vec in my_chunk:
            local_graph.insert_local(vec.tolist())

        print(f"[Node {proxy_port}] ID {node_id}: {len(my_chunk)} Vektoren geladen.")

        if REPLICATION:
            replica_id = (node_id + 1) % NUM_NODES
            replica_start = replica_id * chunk_size
            replica_chunk = dataset[replica_start:replica_start + chunk_size]
            for vec in replica_chunk:
                local_graph.insert_local(vec.tolist())
            print(f"[Node {proxy_port}] Replikat von ID {replica_id}: {len(replica_chunk)} Vektoren geladen.")
    except FileNotFoundError:
        print(f"[Node {proxy_port}] Fehler: dataset.npy nicht gefunden!")

    if bootstrap_port and bootstrap_port != "None":
        local_graph.add_neighbor_edge("127.0.0.1", int(bootstrap_port), [0.0] * 128)

    from protocol import DistributedRouter
    router = DistributedRouter("127.0.0.1", proxy_port)

    asyncio.create_task(router.health_check_loop(local_graph))
    asyncio.create_task(router.start_gossip_loop(local_graph))

    server = grpc.aio.server()

    p2p_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(proxy_port, local_graph, router), server
    )
    
    server.add_insecure_port(f'[::]:{real_port}')

    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    real_p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    b = sys.argv[2] if len(sys.argv) > 2 else None
    n_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    proxy_p = int(sys.argv[4]) if len(sys.argv) > 4 else real_p

    asyncio.run(serve(real_p, b, n_id, proxy_p))