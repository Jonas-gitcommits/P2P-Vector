import asyncio
import grpc
import numpy as np
import p2p_pb2
import p2p_pb2_grpc

async def run_test():
    async with grpc.aio.insecure_channel('127.0.0.1:5000') as channel:
        stub = p2p_pb2_grpc.VectorStoreStub(channel)
        
        query_vector = np.random.rand(128).astype(np.float32)
        vec = p2p_pb2.Vector(values=query_vector.tobytes())
        
        request = p2p_pb2.SearchRequest(query=vec, k=3)
        
        print("Sende Suchanfrage an Port 5000...")
        response = await stub.SearchSimilar(request)
        
        print(f"\nTop Ergebnisse:")
        for i, (p, d) in enumerate(zip(response.nearest_peers, response.distances)):
            print(f"  Node {p.port} (Distanz: {d:.4f})")

if __name__ == "__main__":
    asyncio.run(run_test())