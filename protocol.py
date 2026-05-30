import grpc
import numpy as np
import p2p_pb2
import p2p_pb2_grpc
import asyncio
import random
from config import (
    GOSSIP_INTERVAL_S, HEALTH_CHECK_INTERVAL_S, PING_TIMEOUT_S,
    RPC_TIMEOUT_BASE_S, LATENCY_PRESETS, LATENCY_SCENARIO,
)

class DistributedRouter:
    def __init__(self, my_ip, my_port, rng=None):
        self.my_ip = my_ip
        self.my_port = my_port
        self._channel_pool = {}
        self.rng = rng or random.Random()

    def _get_channel(self, target: str):
        if target not in self._channel_pool:
            self._channel_pool[target] = grpc.aio.insecure_channel(target)
        return self._channel_pool[target]

    async def ask_neighbor_for_vectors(self, target, query_vector, k, ttl, visited_peers,
                                       kth_dist=0.0, fanout_k=0, early_stop_threshold=0.0):
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
            kth_dist=kth_dist,
            fanout_k=fanout_k,
            early_stop_threshold=early_stop_threshold,
        )

        lat_ms, jitter_ms = LATENCY_PRESETS.get(LATENCY_SCENARIO, (0, 0))
        timeout = ttl * (lat_ms + jitter_ms) / 1000 + RPC_TIMEOUT_BASE_S

        try:
            response = await stub.SearchSimilar(request, timeout=timeout)
            return {
                "peers": [
                    (p.ip, p.port, d, gid)
                    for p, d, gid in zip(
                        response.nearest_peers, response.distances, response.vector_ids
                    )
                ],
                "rpc_count": response.rpc_count,
                "visited_nodes": set(response.visited_nodes),
            }
        except grpc.RpcError:
            return {"peers": [], "rpc_count": 0, "visited_nodes": set()}
        
    async def distributed_search(self, local_graph, query_vector, k, ttl, visited_peers,
                                 kth_dist=0.0, fanout_k=0, early_stop_threshold=0.0):
        my_target = f"{self.my_ip}:{self.my_port}"

        if visited_peers is None:
            visited_peers = []

        if my_target not in visited_peers:
            visited_peers.append(my_target)

        if ttl <= 0:
            return {"peers": [], "rpc_count": 0, "visited_nodes": set()}

        from config import ROUTING_FANOUT
        if early_stop_threshold > 0 and kth_dist > 0 and kth_dist <= early_stop_threshold:
            return {"peers": [], "rpc_count": 0, "visited_nodes": set()}

        decision = local_graph.evaluate_next_hop(
            query_vector, visited_peers, fanout=ROUTING_FANOUT
        )
        if decision["action"] == "stop" or not decision["targets"]:
            return {"peers": [], "rpc_count": 0, "visited_nodes": set()}

        targets = decision["targets"][:ROUTING_FANOUT]

        visited_with_siblings = list(set(visited_peers) | set(targets))

        tasks = [
            self.ask_neighbor_for_vectors(
                target, query_vector, k, ttl - 1, visited_with_siblings,
                kth_dist=kth_dist, fanout_k=fanout_k, early_stop_threshold=early_stop_threshold
            )
            for target in targets
        ]

        results = await asyncio.gather(*tasks)

        combined_results = []
        total_rpcs = 0
        all_visited = set()
        for r in results:
            combined_results.extend(r["peers"])
            total_rpcs += r["rpc_count"]
            all_visited |= r["visited_nodes"]

        combined_results.sort(key=lambda x: x[2])
        unique_results = []
        seen = set()
        for ip, port, dist, gid in combined_results:
            if gid not in seen:
                seen.add(gid)
                unique_results.append((ip, port, dist, gid))

        return {"peers": unique_results[:max(fanout_k, k)], "rpc_count": total_rpcs, "visited_nodes": all_visited}

    async def start_gossip_loop(self, local_graph):
        while True:
            await asyncio.sleep(GOSSIP_INTERVAL_S)
            if not local_graph.neighbors:
                continue

            target = self.rng.choice(list(local_graph.neighbors.keys()))
            my_target = f"{self.my_ip}:{self.my_port}"

            probe = None
            for summary in local_graph.neighbors.values():
                if summary is not None:
                    probe = summary[self.rng.randint(0, len(summary) - 1)].tolist()
                    break
            if probe is None:
                probe = [0.0] * local_graph.dimension

            try:
                results = await self.ask_neighbor_for_vectors(
                    target, probe, k=2, ttl=1, visited_peers=[my_target]
                )
                for ip, port, _dist, _gid in results["peers"]:
                    if ip == self.my_ip and port == self.my_port:
                        continue
                    local_graph.add_neighbor_edge(ip, port)
            except Exception:
                pass

    async def health_check_loop(self, local_graph):
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL_S)
            dead_targets = []

            for target in list(local_graph.neighbors.keys()):
                channel = self._get_channel(target)
                stub = p2p_pb2_grpc.VectorStoreStub(channel)

                try:
                    response = await stub.Ping(p2p_pb2.PingRequest(), timeout=PING_TIMEOUT_S)
                    if response.summary_count > 0 and response.summary:
                        local_graph.neighbors[target] = np.frombuffer(
                            response.summary, dtype=np.float32
                        ).reshape(response.summary_count, -1).copy()
                except grpc.RpcError:
                    dead_targets.append(target)

            for target in dead_targets:
                if target in local_graph.neighbors:
                    del local_graph.neighbors[target]
                if target in self._channel_pool:
                    await self._channel_pool[target].close()
                    del self._channel_pool[target]
