import asyncio
import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import faiss
import random
import sys
from config import HNSW_M, DIMENSION, ROUTING_STRATEGY

class LocalGraphState:
    def __init__(self, dimension=DIMENSION, M=HNSW_M, rng=None):
        self.dimension = dimension
        # Angelehnt an [malkov2020hnsw], [douze2024faiss].
        self.local_index = faiss.IndexHNSWFlat(dimension, M)
        self.local_index.hnsw.efSearch = 64
        self.global_ids = []
        self.neighbors = {}
        self.bootstrap_seeds = []
        self.rng = rng or random.Random()
        self._summary_cache = None
        self._summary_lock = asyncio.Lock()

    async def insert_local(self, vector, global_id):
        vec_np = np.array([vector], dtype=np.float32)
        await asyncio.to_thread(self.local_index.add, vec_np)
        self.global_ids.append(global_id)
        self._summary_cache = None

    async def insert_batch(self, vectors, global_ids):
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        await asyncio.to_thread(self.local_index.add, arr)
        self.global_ids.extend(int(g) for g in global_ids)
        self._summary_cache = None

    async def search_local(self, query_vector, k, my_ip, my_port):
        if self.local_index.ntotal == 0:
            return []
        query_np = np.array([query_vector], dtype=np.float32)
        search_k = min(k, self.local_index.ntotal)
        dist, idx = await asyncio.to_thread(self.local_index.search, query_np, search_k)
        results = [
            (my_ip, my_port, float(d), self.global_ids[i])
            for d, i in zip(dist[0], idx[0]) if i >= 0 and np.isfinite(d)
        ]
        results.sort(key=lambda x: x[2])
        return results[:k]

    # Angelehnt an [jegou2011pq].
    async def compute_summary(self, R=8):
        async with self._summary_lock:
            if self._summary_cache is not None:
                return self._summary_cache

            n = self.local_index.ntotal
            if n == 0:
                return b"", 0

            def _run():
                vecs = self.local_index.reconstruct_n(0, n).astype(np.float32)
                if n < R:
                    centroid = vecs.mean(axis=0, keepdims=True).astype(np.float32)
                    return centroid.tobytes(), 1
                kmeans = faiss.Kmeans(self.dimension, R, niter=20, verbose=False)
                kmeans.train(vecs)
                return kmeans.centroids.astype(np.float32).tobytes(), R

            self._summary_cache = await asyncio.to_thread(_run)
            return self._summary_cache

    def evaluate_next_hop(self, query_vector, visited_peers, fanout=2):
        visited_set = set(visited_peers)
        unvisited = [t for t in self.neighbors if t not in visited_set]

        if not unvisited:
            return {"action": "stop", "targets": []}

        # Angelehnt an [lv2002search].
        if ROUTING_STRATEGY == 'flood':
            return {"action": "hop", "targets": unvisited}

        # Angelehnt an [lv2002search].
        if ROUTING_STRATEGY == 'random':
            return {"action": "hop",
                    "targets": self.rng.sample(unvisited, min(fanout, len(unvisited)))}

        # Angelehnt an [malkov2020hnsw].
        query_np = np.array(query_vector, dtype=np.float32)

        def _dist(t):
            s = self.neighbors[t]
            return float("inf") if s is None else float(np.min(np.sum((s - query_np) ** 2, axis=1)))

        return {"action": "hop", "targets": sorted(unvisited, key=_dist)[:fanout]}

    def add_neighbor_edge(self, ip, port):
        target = f"{ip}:{port}"
        if target not in self.neighbors:
            self.neighbors[target] = None  


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
            self.local_graph.add_neighbor_edge(request.sender_ip, request.sender_port)

        is_entry = (request.sender_port == 9999)

        if ROUTING_STRATEGY == 'iterative' and is_entry:
            iter_result = await self.router.iterative_search(
                self.local_graph, query_vec, request.k, request.ttl
            )
            response = p2p_pb2.SearchResponse()
            for ip, port, dist, gid in iter_result["peers"]:
                p = response.nearest_peers.add()
                p.ip = ip
                p.port = port
                response.distances.append(dist)
                response.vector_ids.append(gid)
            response.rpc_count = iter_result["rpc_count"]
            response.visited_nodes.extend(sorted(iter_result["visited_nodes"]))
            return response

        fanout_k = request.fanout_k if request.fanout_k > 0 else max(request.k * 4, 20)
        local_budget = max(fanout_k, request.k)

        local_res = await self.local_graph.search_local(query_vec, local_budget, "127.0.0.1", self.port)
        combined_res = list(local_res)

        if len(local_res) >= request.k:
            kth_dist = float(local_res[request.k - 1][2])
        elif local_res:
            kth_dist = float(local_res[-1][2])
        else:
            kth_dist = request.kth_dist
        my_id = f"127.0.0.1:{self.port}"
        rpc_count = 1
        visited_nodes = {my_id}

        _skip_forward = (request.early_stop_threshold > 0 and bool(local_res)
                         and local_res[0][2] <= request.early_stop_threshold)

        if request.ttl > 0 and not _skip_forward:
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
        for ip, port, dist, gid in combined_res:
            if gid in seen:
                continue
            seen.add(gid)
            deduped.append((ip, port, dist, gid))

        final = deduped[:request.k] if is_entry else deduped[:max(fanout_k, request.k)]

        response = p2p_pb2.SearchResponse()
        for ip, port, dist, gid in final:
            p = response.nearest_peers.add()
            p.ip = ip
            p.port = port
            response.distances.append(dist)
            response.vector_ids.append(gid)
        response.rpc_count = rpc_count
        response.visited_nodes.extend(sorted(visited_nodes))

        return response

    async def QueryNode(self, request, context):
        query_vec = np.frombuffer(request.query.values, dtype=np.float32).tolist() 
        local_res = await self.local_graph.search_local(
            query_vec, request.k, "127.0.0.1", self.port
        )
        response = p2p_pb2.QueryNodeResponse()
        for ip, port, dist, gid in local_res:
            p = response.nearest_peers.add()
            p.ip = ip
            p.port = port
            response.distances.append(dist)
            response.vector_ids.append(gid)
        for target, summary in self.local_graph.neighbors.items():
            nb = response.neighbors.add()
            nb.target = target
            if summary is not None:
                nb.summary = summary.tobytes()
                nb.summary_count = len(summary)
        return response

    async def Ping(self, request, context):
        summary_bytes, summary_count = await self.local_graph.compute_summary()
        return p2p_pb2.PingResponse(
            alive=True,
            neighbor_count=len(self.local_graph.neighbors),
            summary=summary_bytes,
            summary_count=summary_count,
        )

async def serve(real_port, bootstrap_port=None, node_id=0, proxy_port=None, seed=0):
    proxy_port = proxy_port or real_port
    rng = random.Random(seed)
    local_graph = LocalGraphState(rng=rng)

    try:
        from config import VECTORS_PER_NODE, NUM_NODES, REPLICATION, PLACEMENT
        dataset = np.load("dataset.npy", mmap_mode="r")

        if PLACEMENT == 'clustered':
            partition = np.load("partition.npy")
            my_ids = np.where(partition == node_id)[0]
            await local_graph.insert_batch(dataset[my_ids], my_ids.tolist())
            print(f"[Node {proxy_port}] ID {node_id}: {len(my_ids)} Vektoren geladen (clustered).")

            # Angelehnt an [stoica2001chord].
            if REPLICATION:
                replica_id = (node_id + 1) % NUM_NODES
                replica_ids = np.where(partition == replica_id)[0]
                await local_graph.insert_batch(dataset[replica_ids], replica_ids.tolist())
                print(f"[Node {proxy_port}] Replikat von ID {replica_id}: "
                      f"{len(replica_ids)} Vektoren geladen.")
        else:
            chunk_size = VECTORS_PER_NODE
            start_idx = node_id * chunk_size
            if start_idx + chunk_size > len(dataset):
                raise RuntimeError(
                    "dataset.npy zu klein! "
                    "Bitte `python generate_data.py` erneut ausführen."
                )
            my_chunk = dataset[start_idx:start_idx + chunk_size]
            await local_graph.insert_batch(
                my_chunk, list(range(start_idx, start_idx + len(my_chunk))))
            print(f"[Node {proxy_port}] ID {node_id}: {len(my_chunk)} Vektoren geladen (contiguous).")

            # Angelehnt an [stoica2001chord].
            if REPLICATION:
                replica_id = (node_id + 1) % NUM_NODES
                replica_start = replica_id * chunk_size
                replica_chunk = dataset[replica_start:replica_start + chunk_size]
                await local_graph.insert_batch(
                    replica_chunk,
                    list(range(replica_start, replica_start + len(replica_chunk))))
                print(f"[Node {proxy_port}] Replikat von ID {replica_id}: "
                      f"{len(replica_chunk)} Vektoren geladen.")
    except FileNotFoundError as e:
        print(f"[Node {proxy_port}] Fehler: {e}")

    # Angelehnt an [stoica2001chord], [maymounkov2002kademlia].
    if bootstrap_port and bootstrap_port != "None":
        local_graph.add_neighbor_edge("127.0.0.1", int(bootstrap_port))
        local_graph.bootstrap_seeds.append(int(bootstrap_port))

    from protocol import DistributedRouter
    router = DistributedRouter("127.0.0.1", proxy_port, rng=rng)

    asyncio.create_task(router.health_check_loop(local_graph))
    # Angelehnt an [ormandi2013gossip], [demers1987epidemic], [jelasity2007peersampling].
    asyncio.create_task(router.start_gossip_loop(local_graph))

    server = grpc.aio.server()

    p2p_pb2_grpc.add_VectorStoreServicer_to_server(
        VectorStoreServicer(proxy_port, local_graph, router), server
    )
    
    server.add_insecure_port(f'127.0.0.1:{real_port}')

    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    real_p = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    b = sys.argv[2] if len(sys.argv) > 2 else None
    n_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    proxy_p = int(sys.argv[4]) if len(sys.argv) > 4 else real_p
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    try:
        asyncio.run(serve(real_p, b, n_id, proxy_p, seed))
    except (KeyboardInterrupt, SystemExit):
        pass