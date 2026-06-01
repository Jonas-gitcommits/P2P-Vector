import os
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from config import SUBSET_SIZE, NUM_NODES, DIMENSION

_IR_DIM        = 384
_MODEL         = "all-MiniLM-L6-v2"
_BATCH         = 512
_NITER         = 50

_CORPUS_HF     = "BeIR/msmarco"
_N_CORPUS      = 150_000          
_CORPUS_SEED   = 42               

_CACHE_DIR     = "ir_cache"
_CORPUS_CACHE  = os.path.join(_CACHE_DIR, "msmarco_corpus_150k_seed42.npy")
_QUERIES_CACHE = os.path.join(_CACHE_DIR, "msmarco_queries_dev.npy")


def _check_dimension():
    if DIMENSION != _IR_DIM:
        raise RuntimeError(
            f"config.DIMENSION={DIMENSION}, aber der IR-Encoder liefert {_IR_DIM} Dimensionen."
        ) 


def _encode(model, texts, label):
    print(f"Encodiere {len(texts):,} {label}…")
    embs = model.encode(
        texts,
        batch_size=_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   
    )
    return embs.astype(np.float32)


def _verify_norms(embs, label, n=200):
    norms = np.linalg.norm(embs[:n], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"{label}: Normen nicht alle ≈ 1!"


def _spot_check(doc_embs, q_embs):
    probe = doc_embs[:min(50_000, len(doc_embs))]
    idx = faiss.IndexFlatL2(_IR_DIM)
    idx.add(probe)
    dists, nn = idx.search(q_embs[:3], 3)
    for i in range(3):
        print(f"  Query {i} → NN-IDs {nn[i].tolist()}  "
              f"L2²={[round(float(d), 4) for d in dists[i]]}  ")
              


def _build_caches():
    print(f"[Cache-Build] Lade MS MARCO Passagenkorpus...")
    corpus_ds = load_dataset(_CORPUS_HF, "corpus", split="corpus")
    total = len(corpus_ds)
    print(f"  Gesamtkorpus: {total:,} Passagen (Title + Text) laden aus {_CORPUS_HF}.")
    if total < _N_CORPUS:
        raise RuntimeError(
            f"MS MARCO Korpus ({total:,}) kleiner als _N_CORPUS ({_N_CORPUS:,})."
        )

    rng = np.random.default_rng(_CORPUS_SEED)
    indices = rng.choice(total, size=_N_CORPUS, replace=False).tolist()
    doc_texts = [
        f"{row.get('title', '')} {row['text']}".strip()
        for row in corpus_ds.select(indices)
    ]

    print(f"[Cache-Build] Lade MS MARCO Dev-Queries ({_CORPUS_HF})…")
    queries_ds = load_dataset(_CORPUS_HF, "queries", split="queries")
    query_texts = [row["text"] for row in queries_ds]
    print(f"  {len(query_texts):,} Dev-Queries")

    print(f"[Cache-Build] Lade Encoder '{_MODEL}'…")
    model = SentenceTransformer(_MODEL)

    corpus_embs = _encode(model, doc_texts, "MS MARCO Passagen")
    query_embs  = _encode(model, query_texts, "MS MARCO Dev-Queries")

    _verify_norms(corpus_embs, "Korpus-Embeddings")
    _verify_norms(query_embs,  "Query-Embeddings")

    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.save(_CORPUS_CACHE, corpus_embs)
    np.save(_QUERIES_CACHE, query_embs)
    print(f"  Gespeichert: {_CORPUS_CACHE}  {corpus_embs.shape}")
    print(f"  Gespeichert: {_QUERIES_CACHE}  {query_embs.shape}")


def _compute_partition(doc_embs):
    n = len(doc_embs)
    print(f"Berechne Cluster-Partitionierung ({NUM_NODES} Cluster, {n:,} Vektoren, niter={_NITER})…")
    kmeans = faiss.Kmeans(_IR_DIM, NUM_NODES, niter=_NITER, seed=42, verbose=False)
    kmeans.train(doc_embs)

    _, asgn = kmeans.index.search(doc_embs, 1)
    partition = asgn.flatten().astype(np.int32)

    counts = [int(np.sum(partition == i)) for i in range(NUM_NODES)]
    assert sum(counts) == n, f"Partitions-Summe {sum(counts)} ≠ {n}"
    print(f"  Cluster-Größen: min={min(counts)}  max={max(counts)}  Summe={sum(counts)}")

    centroids = kmeans.centroids
    dists = [
        float(np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2)))
        for i in range(NUM_NODES)
        for j in range(i + 1, NUM_NODES)
    ]
    print(f"L2: "
          f"min={min(dists):.3f}  Ø={np.mean(dists):.3f}  max={max(dists):.3f}")
    return partition


def generate_ir_dataset():
    _check_dimension()
    os.makedirs(_CACHE_DIR, exist_ok=True)

    if not (os.path.exists(_CORPUS_CACHE) and os.path.exists(_QUERIES_CACHE)):
        _build_caches()
    else:
        print(f"Lade gecachte MS MARCO Embeddings aus {_CACHE_DIR}/")

    all_corpus = np.load(_CORPUS_CACHE)   
    all_queries = np.load(_QUERIES_CACHE)  
    print(f"  Korpus-Cache: {all_corpus.shape}  Query-Cache: {all_queries.shape}")

    if SUBSET_SIZE > len(all_corpus):
        raise RuntimeError(
            f"SUBSET_SIZE ({SUBSET_SIZE:,}) > Korpus-Cache ({len(all_corpus):,}).\n"
            f"Reduziere NUM_NODES * VECTORS_PER_NODE auf max. {len(all_corpus):,} in config.py."
        )

    doc_embs = all_corpus[:SUBSET_SIZE]   

    _verify_norms(doc_embs,   "Dokumente (Subset)")
    _verify_norms(all_queries, "Queries")
    _spot_check(doc_embs, all_queries)

    np.save("dataset.npy",  doc_embs)
    np.save("queries.npy",  all_queries)
    print(f"  dataset.npy:   {doc_embs.shape}")
    print(f"  queries.npy:   {all_queries.shape}")

    partition = _compute_partition(doc_embs)
    np.save("partition.npy", partition)
    print(f"  partition.npy: {partition.shape}  dtype={partition.dtype}")


if __name__ == "__main__":
    generate_ir_dataset()
