import numpy as np
import os
import sys
import faiss
from config import SUBSET_SIZE, NUM_NODES, DATASET, IR_CORPUS_CACHE, IR_QUERIES_CACHE

SIFT_DIR = "sift_data/sift"


# Übernommen aus facebookresearch/faiss, contrib/vecs_io.py [douze2024faiss].
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    if sys.byteorder == 'big':
        a.byteswap(inplace=True)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def ivecs_mmap(fname):
    assert sys.byteorder != 'big'
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')


def read_fvecs(path, max_count=None):
    vecs = fvecs_mmap(path)
    if max_count is not None:
        vecs = vecs[:max_count]
    return np.ascontiguousarray(vecs, dtype=np.float32)


def _verify_norms(embs, label, n=200):
    norms = np.linalg.norm(embs[:n], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"{label}: Normen nicht alle ≈ 1!"


def _compute_partition(embs):
    n, dim = len(embs), embs.shape[1]
    print(f"Berechne Cluster-Partitionierung ({NUM_NODES} Cluster, {n:,} Vektoren, niter=50)...")
    kmeans = faiss.Kmeans(dim, NUM_NODES, niter=50, seed=42, verbose=False)
    kmeans.train(embs)

    _, assignments = kmeans.index.search(embs, 1)
    partition = assignments.flatten().astype(np.int32)

    counts = [int(np.sum(partition == i)) for i in range(NUM_NODES)]
    assert sum(counts) == n, f"Partitions-Summe {sum(counts)} != {n}"
    print(f"  Cluster-Größen: min={min(counts)}  max={max(counts)}  Summe={sum(counts)}")

    centroids = kmeans.centroids
    dists = [
        float(np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2)))
        for i in range(NUM_NODES)
        for j in range(i + 1, NUM_NODES)
    ]
    print(f"  Paarweise Zentroid-Distanzen (L2):  "
          f"min={min(dists):.3f}  Ø={np.mean(dists):.3f}  max={max(dists):.3f}")

    return partition


def _generate_sift():
    base_path  = os.path.join(SIFT_DIR, "sift_base.fvecs")
    query_path = os.path.join(SIFT_DIR, "sift_query.fvecs")

    for p in [base_path, query_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} fehlt. Erst python download_sift.py ausführen.")

    print(f"Lade SIFT1M-Subset: erste {SUBSET_SIZE} Vektoren...")
    base    = read_fvecs(base_path, max_count=SUBSET_SIZE)
    queries = read_fvecs(query_path)

    np.save("dataset.npy", base)
    np.save("queries.npy", queries)
    print(f"  dataset.npy:   {base.shape}")
    print(f"  queries.npy:   {queries.shape}")

    partition = _compute_partition(base)
    np.save("partition.npy", partition)
    print(f"  partition.npy: {partition.shape}  (dtype={partition.dtype})")


def _generate_ir():
    for p in [IR_CORPUS_CACHE, IR_QUERIES_CACHE]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} fehlt. Erst python download_ir.py ausführen.")

    print("Lade IR-Cache...")
    all_corpus  = np.load(IR_CORPUS_CACHE)
    all_queries = np.load(IR_QUERIES_CACHE)
    print(f"  Korpus-Cache: {all_corpus.shape}  Query-Cache: {all_queries.shape}")

    if SUBSET_SIZE > len(all_corpus):
        raise RuntimeError(
            f"SUBSET_SIZE ({SUBSET_SIZE:,}) > Korpus-Cache ({len(all_corpus):,}). "
            f"Reduziere NUM_NODES * VECTORS_PER_NODE auf max. {len(all_corpus):,} in config.py."
        )

    doc_embs = all_corpus[:SUBSET_SIZE]
    _verify_norms(doc_embs,   "Dokumente (Subset)")
    _verify_norms(all_queries, "Queries")

    np.save("dataset.npy",  doc_embs)
    np.save("queries.npy",  all_queries)
    print(f"  dataset.npy:   {doc_embs.shape}")
    print(f"  queries.npy:   {all_queries.shape}")

    partition = _compute_partition(doc_embs)
    np.save("partition.npy", partition)
    print(f"  partition.npy: {partition.shape}  dtype={partition.dtype}")


def generate_dataset():
    if DATASET == 'ir':
        _generate_ir()
    else:
        _generate_sift()


if __name__ == "__main__":
    generate_dataset()
