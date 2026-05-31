import numpy as np
import os
import struct
import faiss
from config import SUBSET_SIZE, NUM_NODES, DIMENSION

SIFT_DIR = "sift_data/sift"


def read_fvecs(path, max_count=None):
    vecs = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            vecs.append(vec)
            if max_count is not None and len(vecs) >= max_count:
                break
    return np.array(vecs, dtype=np.float32)


def read_ivecs(path, max_count=None):
    vecs = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("<i", dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
            vecs.append(vec)
            if max_count is not None and len(vecs) >= max_count:
                break
    return np.array(vecs, dtype=np.int32)


def _compute_partition(base):
    print(f"Berechne Cluster-Partitionierung ({NUM_NODES} Cluster, niter=50)...")
    kmeans = faiss.Kmeans(DIMENSION, NUM_NODES, niter=50, seed=42, verbose=False)
    kmeans.train(base)

    _, assignments = kmeans.index.search(base, 1)
    partition = assignments.flatten().astype(np.int32)

    counts = [int(np.sum(partition == i)) for i in range(NUM_NODES)]
    assert sum(counts) == SUBSET_SIZE, f"Summe {sum(counts)} != SUBSET_SIZE {SUBSET_SIZE}"
    print(f"  Cluster-Größen: {counts}  (Summe={sum(counts)} = SUBSET_SIZE)")

    centroids = kmeans.centroids
    dists = []
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            d = float(np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2)))
            dists.append(d)
    print(f"  Paarweise Zentroid-Distanzen (L2):  "
          f"min={min(dists):.1f}  Ø={np.mean(dists):.1f}  max={max(dists):.1f}")

    return partition


def generate_dataset():
    base_path  = os.path.join(SIFT_DIR, "sift_base.fvecs")
    query_path = os.path.join(SIFT_DIR, "sift_query.fvecs")
    gt_path    = os.path.join(SIFT_DIR, "sift_groundtruth.ivecs")

    for p in [base_path, query_path, gt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} fehlt. Erst python download_sift.py ausführen."
            )

    print(f"Lade SIFT1M-Subset: erste {SUBSET_SIZE} Vektoren...")
    base    = read_fvecs(base_path, max_count=SUBSET_SIZE)
    queries = read_fvecs(query_path)
    gt      = read_ivecs(gt_path)

    np.save("dataset.npy", base)
    np.save("queries.npy", queries)
    np.save("ground_truth.npy", gt)

    print(f"  dataset.npy:      {base.shape}")
    print(f"  queries.npy:      {queries.shape}")
    print(f"  ground_truth.npy: {gt.shape}")

    partition = _compute_partition(base)
    np.save("partition.npy", partition)
    print(f"  partition.npy:    {partition.shape}  (dtype={partition.dtype})")


if __name__ == "__main__":
    generate_dataset()
