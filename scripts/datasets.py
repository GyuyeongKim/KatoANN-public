#!/usr/bin/env python3
"""Download and convert the evaluation datasets to base.fbin, query.fbin, gt.bin.

  openai3072-1m  1M DBpedia entities embedded with OpenAI text-embedding-3-large
                 (3,072 dimensions), from Qdrant on Hugging Face. 10,000 rows are
                 held out as queries with a NumPy permutation (seed 42); the other
                 990,000 are the base. Ground truth is computed with faiss.
  bioasq-10m     10M BioASQ PubMed abstracts (1,024 dimensions) and their 3,106
                 test queries from VectorDBBench, with its ground truth.

Vectors are normalized to unit length (cosine). Each .fbin holds a uint32 count,
a uint32 dimension, and float32 rows; gt.bin holds a uint32 count, a uint32 k,
and the uint32 ids of the k nearest base vectors per query.
usage: datasets.py NAME OUT_DIR
"""
import os
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq


def write_fbin(path, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    with open(path + ".tmp", "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.tofile(f)
    os.replace(path + ".tmp", path)


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    x /= n


def embedding_column(table, name):
    col = table.column(name).combine_chunks()
    flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32)
    return flat.reshape(len(col), len(flat) // len(col))


def faiss_gt(base, query, k=100):
    import faiss
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    return index.search(query, k)[1].astype(np.uint32)


def write_gt(path, ids):
    with open(path, "wb") as f:
        np.array(ids.shape, dtype=np.uint32).tofile(f)
        ids.astype(np.uint32).tofile(f)


def openai3072_1m(out):
    from huggingface_hub import snapshot_download
    repo = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
    src = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.parquet"],
                            local_dir=os.path.join(out, "download"))
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(src) for f in fs
                   if f.endswith(".parquet"))
    col = "text-embedding-3-large-3072-embedding"
    x = np.vstack([embedding_column(pq.read_table(f, columns=[col]), col) for f in files])
    normalize(x)
    perm = np.random.default_rng(42).permutation(len(x))
    base, query = x[perm[10000:]], x[perm[:10000]]
    write_fbin(os.path.join(out, "base.fbin"), base)
    write_fbin(os.path.join(out, "query.fbin"), query)
    write_gt(os.path.join(out, "gt.bin"), faiss_gt(base, query))


def bioasq_10m(out):
    url = "https://assets.zilliz.com/benchmark/bioasq_large_10m"
    dl = os.path.join(out, "download")
    os.makedirs(dl, exist_ok=True)
    names = ["test.parquet", "neighbors.parquet"] + \
            [f"shuffle_train-{i:02d}-of-10.parquet" for i in range(10)]
    for name in names:
        path = os.path.join(dl, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{url}/{name}", path + ".tmp")
            os.replace(path + ".tmp", path)

    def read(names):
        ids, embs = [], []
        for name in names:
            t = pq.read_table(os.path.join(dl, name), columns=["id", "emb"])
            ids.append(t.column("id").to_numpy())
            embs.append(embedding_column(t, "emb"))
        ids, x = np.concatenate(ids), np.vstack(embs)
        x = x[np.argsort(ids, kind="stable")]
        normalize(x)
        return x

    write_fbin(os.path.join(out, "base.fbin"), read(names[2:]))
    write_fbin(os.path.join(out, "query.fbin"), read(names[:1]))
    t = pq.read_table(os.path.join(dl, "neighbors.parquet"))
    order = np.argsort(t.column("id").to_numpy(), kind="stable")
    nb = t.column("neighbors_id").to_pylist()
    write_gt(os.path.join(out, "gt.bin"), np.array([nb[i][:100] for i in order], dtype=np.uint32))


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("openai3072-1m", "bioasq-10m"):
        sys.exit(__doc__)
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    {"openai3072-1m": openai3072_1m, "bioasq-10m": bioasq_10m}[sys.argv[1]](out)
    print(f"{sys.argv[1]}: wrote base.fbin, query.fbin, gt.bin to {out}")


if __name__ == "__main__":
    main()
