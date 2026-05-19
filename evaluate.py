import grpc
import p2p_pb2
import p2p_pb2_grpc
import numpy as np
import time
import random
from config import NUM_NODES

NUM_QUERIES = 100
K = 3
TTL = 4
ALL_NODES = [f"127.0.0.1:{5001 + i}" for i in range(NUM_NODES)]


def test_search():
    queries = np.load("queries.npy")
    dataset = np.load("dataset.npy")
    full_gt = np.load("ground_truth.npy")

    random.seed(42)
    latencies = []
    total_recall = 0.0

    for qi in range(NUM_QUERIES):
        query_vec = queries[qi]
        gt_indices = full_gt[qi][:K]
        gt_vecs = dataset[gt_indices]
        diffs = gt_vecs - query_vec
        gt_dists = np.sum(diffs ** 2, axis=1)
        gt_max = float(np.max(gt_dists))

        entry_node = random.choice(ALL_NODES)
        channel = grpc.insecure_channel(entry_node)
        stub = p2p_pb2_grpc.VectorStoreStub(channel)

        query_bytes = np.array(query_vec, dtype=np.float32).tobytes()
        req = p2p_pb2.SearchRequest(
            query=p2p_pb2.Vector(values=query_bytes),
            k=K,
            ttl=TTL,
            visited_peers=[],
            sender_ip="127.0.0.1",
            sender_port=9999
        )

        t0 = time.time()
        res = stub.SearchSimilar(req)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        matches = sum(1 for d in res.distances if d <= gt_max + 1e-5)
        recall = min(matches, K) / K
        total_recall += recall

        channel.close()

    avg_recall = (total_recall / NUM_QUERIES) * 100
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"Recall: {avg_recall:.2f} %, Avg Latenz: {avg_latency:.2f} ms, P95 Latenz: {p95_latency:.2f} ms")


if __name__ == '__main__':
    test_search()
