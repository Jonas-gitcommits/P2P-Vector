import os
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from config import IR_CACHE_DIR, IR_CORPUS_CACHE, IR_QUERIES_CACHE, DIMENSION

_IR_DIM      = 384
_MODEL       = "all-MiniLM-L6-v2"
_BATCH       = 512
_N_CORPUS    = 200_000
_N_QUERIES   = 7_000
_CORPUS_SEED = 42
_CORPUS_HF   = "BeIR/msmarco"


def _check_dimension():
    if DIMENSION != _IR_DIM:
        raise RuntimeError(
            f"Konfigurationsfehler: DIMENSION ({DIMENSION}) stimmt nicht mit IR_DIM ({_IR_DIM}) überein. "
        )


def _encode(model, texts, label):
    print(f"Encodiere {len(texts):,} {label}…")
    return model.encode(
        texts,
        batch_size=_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def _verify_norms(embs, label, n=200):
    norms = np.linalg.norm(embs[:n], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"{label}: Normen nicht auf 1"
 
def _spot_check(doc_embs, q_embs):
    probe = doc_embs[:min(50_000, len(doc_embs))]
    idx = faiss.IndexFlatL2(_IR_DIM)
    idx.add(probe)
    dists, nn = idx.search(q_embs[:3], 3)
    for i in range(3):
        print(f"  Query {i} → NN-IDs {nn[i].tolist()}  "
              f"L2²={[round(float(d), 4) for d in dists[i]]}")


def _build_caches():
    print("[Cache-Build] Lade MS MARCO Passagenkorpus...")
    corpus_ds = load_dataset(_CORPUS_HF, "corpus", split="corpus")
    total = len(corpus_ds)
    print(f"  Gesamtkorpus: {total:,} Passagen aus {_CORPUS_HF}.")
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

    print(f"[Cache-Build] Lade MS MARCO Queries ({_CORPUS_HF})…")
    queries_ds = load_dataset(_CORPUS_HF, "queries", split="queries")
    rng_q = np.random.default_rng(_CORPUS_SEED + 1)
    q_indices = sorted(
        rng_q.choice(len(queries_ds), size=min(_N_QUERIES, len(queries_ds)), replace=False).tolist()
    )
    query_texts = [queries_ds[int(i)]["text"] for i in q_indices]
    print(f"  {len(query_texts):,} Queries (sample aus {len(queries_ds):,}, seed={_CORPUS_SEED + 1})")

    print(f"[Cache-Build] Lade Encoder '{_MODEL}'…")
    model = SentenceTransformer(_MODEL)

    corpus_embs = _encode(model, doc_texts, "MS MARCO Passagen")
    query_embs  = _encode(model, query_texts, "MS MARCO Dev-Queries")

    _verify_norms(corpus_embs, "Korpus-Embeddings")
    _verify_norms(query_embs,  "Query-Embeddings")

    os.makedirs(IR_CACHE_DIR, exist_ok=True)
    np.save(IR_CORPUS_CACHE, corpus_embs)
    np.save(IR_QUERIES_CACHE, query_embs)
    print(f"  Gespeichert: {IR_CORPUS_CACHE}  {corpus_embs.shape}")
    print(f"  Gespeichert: {IR_QUERIES_CACHE}  {query_embs.shape}")

    _spot_check(corpus_embs, query_embs)


def ensure_caches():
    _check_dimension()
    os.makedirs(IR_CACHE_DIR, exist_ok=True)
    if os.path.exists(IR_CORPUS_CACHE) and os.path.exists(IR_QUERIES_CACHE):
        print(f"[IR] Cache bereits vorhanden in {IR_CACHE_DIR}/  (übersprungen)")
        return
    _build_caches()


if __name__ == "__main__":
    ensure_caches()
