import grpc
import numpy as np
import p2p_pb2
import p2p_pb2_grpc
import asyncio

class DistributedRouter:
    def __init__(self, my_ip, my_port):
        self.my_ip = my_ip
        self.my_port = my_port
        self._channel_pool = {}

    def _get_channel(self, target: str):
        if target not in self._channel_pool:
            self._channel_pool[target] = grpc.aio.insecure_channel(target)
        return self._channel_pool[target]

    async def ask_neighbor_for_vectors(self, target, query_vector, k, ttl, visited_peers):
        channel = self._get_channel(target)
        stub = p2p_pb2_grpc.VectorStoreStub(channel)
        
        query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
        vec = p2p_pb2.Vector(values=query_bytes)
        
        request = p2p_pb2.SearchRequest(
            query=vec, 
            k=k, 
            ttl=ttl, 
            visited_peers=visited_peers
        )
        
        try:
            response = await stub.SearchSimilar(request, timeout=1.5)
            return [(p.ip, p.port, d) for p, d in zip(response.nearest_peers, response.distances)]
        except grpc.RpcError:
            return [] 
        
    async def distributed_search(self, neighbors, query_vector, k, ttl, visited_peers):
        my_target = f"{self.my_ip}:{self.my_port}"
        
        if visited_peers is None:
            visited_peers = []
            
        if my_target not in visited_peers:
            visited_peers.append(my_target)

        if ttl <= 0:
            return []

        tasks = []
        for target in neighbors:
            if target in visited_peers:
                continue 
            
            task = self.ask_neighbor_for_vectors(
                target, query_vector, k, ttl - 1, list(visited_peers)
            )
            tasks.append(task)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)

        combined_results = []
        for res_list in results:
            combined_results.extend(res_list)

        combined_results.sort(key=lambda x: x[2])
        unique_results = []
        seen = set()
        for ip, port, dist in combined_results:
            key = (ip, port, round(dist, 5))
            if key not in seen:
                seen.add(key)
                unique_results.append((ip, port, dist))

        return unique_results[:k]