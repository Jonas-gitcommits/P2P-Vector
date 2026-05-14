import numpy as np
import os
import struct
from config import SUBSET_SIZE

SIFT_DIR = "sift_data/sift"


def read_fvecs(path, max_count=None):
    """Liest .fvecs-Format (Standard für SIFT/GIST-Benchmarks).

    Format pro Vektor: 4-Byte int32 dim, dann dim × 4-Byte float32.
    Hier: http://corpus-texmex.irisa.fr/
    """
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
    """Liest .ivecs-Format (für Ground-Truth)."""
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


if __name__ == "__main__":
    generate_dataset()
