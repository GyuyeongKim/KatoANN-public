#!/usr/bin/env python3
"""Write a synthetic embedding-like dataset: base.fbin, query.fbin, gt.bin.

Vectors are unit-length, grouped around random centers, with variance that
decays along the dimensions as in text embeddings. The ground truth is the
exact top-100 by squared L2.
usage: make_synthetic.py OUT_DIR [--n 200000] [--dim 768] [--queries 1000]
"""
import argparse
import os

import numpy as np


def write_fbin(path, x):
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.astype(np.float32).tofile(f)


def make(rng, n, dim, centers, scale):
    x = centers[rng.integers(0, len(centers), n)] + 0.5 * rng.standard_normal((n, dim), np.float32)
    x *= scale
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    centers = rng.standard_normal((1000, a.dim), np.float32)
    scale = np.exp(-np.arange(a.dim, dtype=np.float32) / (a.dim / 6))
    base = make(rng, a.n, a.dim, centers, scale)
    query = make(rng, a.queries, a.dim, centers, scale)
    write_fbin(os.path.join(a.out, "base.fbin"), base)
    write_fbin(os.path.join(a.out, "query.fbin"), query)
    k = 100
    gt = np.empty((a.queries, k), np.uint32)
    bn = (base * base).sum(1)
    for s in range(0, a.queries, 256):
        q = query[s:s + 256]
        d = bn[None, :] - 2 * q @ base.T
        top = np.argpartition(d, k, axis=1)[:, :k]
        order = np.take_along_axis(d, top, 1).argsort(1)
        gt[s:s + 256] = np.take_along_axis(top, order, 1)
    with open(os.path.join(a.out, "gt.bin"), "wb") as f:
        np.array([a.queries, k], dtype=np.uint32).tofile(f)
        gt.tofile(f)
    print(f"wrote {a.n} base vectors and {a.queries} queries of dimension {a.dim} to {a.out}")


if __name__ == "__main__":
    main()
