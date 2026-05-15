import grpc
import numpy as np
import p2p_pb2
import p2p_pb2_grpc

class DistributedRouter:
    async def ask_neighbor(self, neighbor_port, query_vector, k):
        """Fragt exakt eine festen Nachbarn nach seinen Ergebnissen."""
        target = f"127.0.0.1:{neighbor_port}"
        
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                stub = p2p_pb2_grpc.VectorStoreStub(channel)
                
                query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
                vec = p2p_pb2.Vector(values=query_bytes)
                
                request = p2p_pb2.SearchRequest(query=vec, k=k)
                
                response = await stub.SearchSimilar(request, timeout=1.5)
                return [(p.ip, p.port, d) for p, d in zip(response.nearest_peers, response.distances)]
        except Exception as e:
            print(f"[Router] Nachbar auf Port {neighbor_port} antwortet nicht.")
            return []