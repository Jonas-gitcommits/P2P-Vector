import grpc
import numpy as np
import p2p_pb2
import p2p_pb2_grpc
import asyncio
import random

class DistributedRouter:
    def __init__(self, my_ip, my_port):
        self.my_ip = my_ip
        self.my_port = my_port
        self._channel_pool = {}

    def _get_channel(self, target: str):
        if target not in self._channel_pool:
            self._channel_pool[target] = grpc.aio.insecure_channel(target)
        return self._channel_pool[target]

    async def ask_neighbor_for_vectors(self, target, query_vector, k, ttl, visited_peers,
                                       best_dist_so_far=0.0, fanout_k=0):
        channel = self._get_channel(target)
        stub = p2p_pb2_grpc.VectorStoreStub(channel)

        query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        vec = p2p_pb2.Vector(values=query_bytes)

        request = p2p_pb2.SearchRequest(
            query=vec,
            k=k,
            ttl=ttl,
            visited_peers=list(visited_peers),
            sender_ip=self.my_ip,
            sender_port=self.my_port,
            best_dist_so_far=best_dist_so_far,
            fanout_k=fanout_k,
        )

        try:
            response = await stub.SearchSimilar(request, timeout=1.5)
            return [(p.ip, p.port, d) for p, d in zip(response.nearest_peers, response.distances)]
        except grpc.RpcError:
            return []
        
    async def distributed_search(self, local_graph, query_vector, k, ttl, visited_peers,
                                 best_dist_so_far=0.0, fanout_k=0):
        my_target = f"{self.my_ip}:{self.my_port}"

        if visited_peers is None:
            visited_peers = []

        if my_target not in visited_peers:
            visited_peers.append(my_target)

        if ttl <= 0:
            return []

        from config import EARLY_STOP_ENABLED, EARLY_STOP_THRESHOLD, ROUTING_FANOUT
        if EARLY_STOP_ENABLED and best_dist_so_far > 0 and best_dist_so_far <= EARLY_STOP_THRESHOLD:
            return []

        decision = local_graph.evaluate_next_hop(
            query_vector, visited_peers, fanout=ROUTING_FANOUT
        )
        if decision["action"] == "stop" or not decision["targets"]:
            return []

        targets = decision["targets"][:ROUTING_FANOUT]

        tasks = [
            self.ask_neighbor_for_vectors(
                target, query_vector, k, ttl - 1, list(visited_peers),
                best_dist_so_far=best_dist_so_far, fanout_k=fanout_k
            )
            for target in targets
        ]

        results = await asyncio.gather(*tasks)

        combined_results = []
        for res_list in results:
            combined_results.extend(res_list)

        combined_results.sort(key=lambda x: x[2])
        unique_results = []
        seen = set()
        for ip, port, dist in combined_results:
            key = round(dist, 5)
            if key not in seen:
                seen.add(key)
                unique_results.append((ip, port, dist))

        return unique_results[:max(fanout_k, k)]

    async def start_gossip_loop(self, local_graph):
        while True:
            await asyncio.sleep(5)
            if not local_graph.neighbors:
                continue

            target = random.choice(list(local_graph.neighbors.keys()))
            my_vector = local_graph.get_my_latest_vector()
            my_target = f"{self.my_ip}:{self.my_port}"

            try:
                results = await self.ask_neighbor_for_vectors(
                    target, my_vector, k=2, ttl=1, visited_peers=[my_target]
                )
                for ip, port, _ in results:
                    if ip == self.my_ip and port == self.my_port:
                        continue
                    local_graph.add_neighbor_edge(ip, port, my_vector)
            except Exception:
                pass
    
    async def health_check_loop(self, local_graph):
        while True:
            await asyncio.sleep(5)
            dead_targets = []
            
            for target in list(local_graph.neighbors.keys()):
                channel = self._get_channel(target)
                stub = p2p_pb2_grpc.VectorStoreStub(channel)
                
                try:
                    await stub.Ping(p2p_pb2.PingRequest(), timeout=1.0)
                except grpc.RpcError:
                    dead_targets.append(target)
            
            for target in dead_targets:
                if target in local_graph.neighbors:
                    del local_graph.neighbors[target]
                if target in self._channel_pool:
                    await self._channel_pool[target].close()
                    del self._channel_pool[target]
