#!/usr/bin/env python3
"""Build and measure the baseline systems with the protocol of `katoann search`
and `katoann load`: the same query batches, threads, recall@k, and JSON lines.

  faiss-ivf    Faiss IndexIVFFlat, float32 scan of the probed lists (param: nprobe)
  hnsw         hnswlib HNSW graph, M 32, ef_construction 200 (param: ef)
  svs-leanvec  Intel SVS Vamana graph over LeanVec: the graph search reads
               8-bit LVQ codes of a PCA projection (--dims) and reranks with
               float16 copies of the vectors; degree 64, build window 200
               (param: search window)
  scann        Google ScaNN: a k-means tree (--nlist leaves), anisotropic
               4-bit codes (2 dimensions per code) scanned in registers, and
               exact float32 reordering (param: leaves to search, or
               leaves:reorder with reorder 200 when omitted)

usage:
  bench.py build  --system S --data BASE.fbin --out DIR [--nlist N] [--dims D] [--threads T]
  bench.py search --system S --index DIR --queries Q.fbin --gt GT.bin [--k 10]
                  --param 64,128 [--batch 4096] [--repeat 3] [--threads T]
  bench.py load   --system S --index DIR --queries Q.fbin --param P --rate QPS
                  [--requests 10000] [--max-batch 4096] [--seed 42] [--threads T]
The vectors are unit-length, so inner product and L2 rank alike.
"""
import argparse
import json
import os
import time

import numpy as np


def read_fbin(path):
    n, d = np.fromfile(path, dtype=np.uint32, count=2)
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(int(n), int(d))


def read_gt(path):
    g = np.fromfile(path, dtype=np.uint32)
    return g[2:].reshape(int(g[0]), int(g[1]))


class FaissIVF:
    def __init__(self, threads):
        import faiss
        self.faiss = faiss
        faiss.omp_set_num_threads(threads)

    def build(self, x, out, a):
        f = self.faiss
        q = f.IndexFlatL2(x.shape[1])
        index = f.IndexIVFFlat(q, x.shape[1], a.nlist, f.METRIC_L2)
        rng = np.random.default_rng(1)
        index.train(x[rng.choice(len(x), min(len(x), 256000), replace=False)])
        for s in range(0, len(x), 1_000_000):
            index.add(x[s:s + 1_000_000])
        f.write_index(index, os.path.join(out, "faiss_ivf.index"))

    def load(self, d):
        self.index = self.faiss.read_index(os.path.join(d, "faiss_ivf.index"))

    def search(self, q, k, p):
        self.index.nprobe = p
        return self.index.search(q, k)[1]


class HNSW:
    def __init__(self, threads):
        import hnswlib
        self.hnswlib, self.threads = hnswlib, threads

    def build(self, x, out, a):
        index = self.hnswlib.Index(space="l2", dim=x.shape[1])
        index.init_index(max_elements=len(x), M=32, ef_construction=200, random_seed=1)
        index.set_num_threads(self.threads)
        for s in range(0, len(x), 1_000_000):
            index.add_items(x[s:s + 1_000_000], np.arange(s, min(len(x), s + 1_000_000)))
        index.save_index(os.path.join(out, "hnsw.bin"))

    def load(self, d):
        dim = int(open(os.path.join(d, "dim")).read())
        self.index = self.hnswlib.Index(space="l2", dim=dim)
        self.index.load_index(os.path.join(d, "hnsw.bin"))
        self.index.set_num_threads(self.threads)

    def search(self, q, k, p):
        self.index.set_ef(max(p, k))
        return self.index.knn_query(q, k=k, num_threads=self.threads)[0]


class SVSLeanVec:
    def __init__(self, threads):
        import svs
        self.svs, self.threads = svs, threads

    def build(self, x, out, a):
        svs = self.svs
        # SVS reads the .fbin layout as a ".bin" file.
        src = os.path.join(out, "base.bin")
        if not os.path.exists(src):
            os.symlink(os.path.abspath(a.data), src)
        data = svs.LeanVecLoader(svs.VectorDataLoader(src, svs.DataType.float32), a.dims,
                                 primary_kind=svs.LeanVecKind.lvq8,
                                 secondary_kind=svs.LeanVecKind.float16)
        params = svs.VamanaBuildParameters(graph_max_degree=64, window_size=200, alpha=0.95,
                                           max_candidate_pool_size=600, prune_to=60,
                                           use_full_search_history=True)
        index = svs.Vamana.build(params, data, svs.DistanceType.MIP, num_threads=self.threads)
        index.save(os.path.join(out, "config"), os.path.join(out, "graph"), os.path.join(out, "data"))

    def load(self, d):
        svs = self.svs
        self.index = svs.Vamana(os.path.join(d, "config"), svs.GraphLoader(os.path.join(d, "graph")),
                                svs.LeanVecLoader(os.path.join(d, "data")), svs.DistanceType.MIP,
                                num_threads=self.threads)

    def search(self, q, k, p):
        self.index.search_window_size = p
        return self.index.search(q, k)[0]


class ScaNN:
    def __init__(self, threads):
        import scann
        self.scann, self.threads = scann, threads

    @staticmethod
    def parse(p):
        leaves, _, reorder = str(p).partition(":")
        return int(leaves), int(reorder or 200)

    def build(self, x, out, a):
        b = (self.scann.scann_ops_pybind.builder(x, 10, "dot_product")
             .tree(num_leaves=a.nlist, num_leaves_to_search=max(1, a.nlist // 8),
                   training_sample_size=min(len(x), 250000))
             .score_ah(2, anisotropic_quantization_threshold=0.2)
             .reorder(200))
        if hasattr(b, "set_n_training_threads"):
            b = b.set_n_training_threads(self.threads)
        # Asset paths are stored relative to the index directory, so the index can move;
        # load_searcher joins them to the directory it is given.
        b.build().serialize(os.path.abspath(out), relative_path=True)

    def load(self, d):
        self.index = self.scann.scann_ops_pybind.load_searcher(os.path.abspath(d))

    def search(self, q, k, p):
        leaves, reorder = self.parse(p)
        return self.index.search_batched_parallel(q, final_num_neighbors=k, leaves_to_search=leaves,
                                                  pre_reorder_num_neighbors=max(k, reorder))[0]


SYSTEMS = {"faiss-ivf": FaissIVF, "hnsw": HNSW, "svs-leanvec": SVSLeanVec, "scann": ScaNN}


def param_value(v):
    """An integer parameter, or a composite one such as ScaNN's leaves:reorder."""
    return int(v) if v.isdigit() else v


def recall(ids, gt, k):
    return float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(ids, gt)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search", "load"])
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--index")
    ap.add_argument("--queries")
    ap.add_argument("--gt")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--param", default="64")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--dims", type=int, default=256)
    ap.add_argument("--rate", type=float, default=1000)
    ap.add_argument("--requests", type=int, default=10000)
    ap.add_argument("--max-batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sysobj = SYSTEMS[a.system](a.threads)

    if a.cmd == "build":
        x = read_fbin(a.data)
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "dim"), "w").write(str(x.shape[1]))
        t0 = time.time()
        sysobj.build(x, a.out, a)
        print(json.dumps({"system": a.system, "vectors": len(x), "build_s": round(time.time() - t0, 1)}))
        return

    t0 = time.time()
    sysobj.load(a.index)
    q = np.ascontiguousarray(read_fbin(a.queries))
    print(json.dumps({"system": a.system, "threads": a.threads, "load_s": round(time.time() - t0, 1),
                      "queries": len(q)}), flush=True)

    if a.cmd == "search":
        gt = read_gt(a.gt) if a.gt else None
        for p in [param_value(v) for v in a.param.split(",")]:
            def run():
                return np.vstack([sysobj.search(q[s:s + a.batch], a.k, p)
                                  for s in range(0, len(q), a.batch)])
            t0 = time.perf_counter()
            ids = run()
            secs = [time.perf_counter() - t0]
            # Below 10 QPS the first pass is the one measurement, and the sweep ends.
            slow = len(q) / secs[0] < 10
            if not slow:
                secs = []
                for _ in range(a.repeat):
                    t0 = time.perf_counter()
                    ids = run()
                    secs.append(time.perf_counter() - t0)
            med = sorted(secs)[len(secs) // 2]
            row = {"param": p, "k": a.k, "batch": a.batch, "qps": round(len(q) / med, 1),
                   "seconds_median": round(med, 4), "rounds": len(secs)}
            if gt is not None:
                row["recall"] = round(recall(ids, gt, a.k), 5)
            print(json.dumps(row), flush=True)
            if slow:
                break
        return

    p = param_value(a.param)
    rng = np.random.default_rng(a.seed)
    arrival = np.cumsum(rng.exponential(1 / a.rate, a.requests))
    which = np.resize(np.arange(len(q)), a.requests)
    sysobj.search(q[:min(len(q), a.max_batch)], a.k, p)
    latency = np.empty(a.requests)
    sizes = []
    start = time.perf_counter()
    done = 0
    while done < a.requests:
        now = time.perf_counter() - start
        if arrival[done] > now:
            time.sleep(arrival[done] - now)
            now = time.perf_counter() - start
        end = min(a.requests, done + a.max_batch, int(np.searchsorted(arrival, now, side="right")))
        end = max(end, done + 1)
        sysobj.search(q[which[done:end]], a.k, p)
        t = time.perf_counter() - start
        latency[done:end] = t - arrival[done:end]
        sizes.append(end - done)
        done = end
    span = time.perf_counter() - start
    lat = np.sort(latency)
    pct = lambda v: float(lat[min(len(lat) - 1, int(np.ceil(v * len(lat))) - 1)] * 1e3)
    print(json.dumps({"system": a.system, "rate": a.rate, "requests": a.requests, "param": p,
                      "achieved_qps": round(a.requests / span, 1), "p50_ms": round(pct(0.5), 2),
                      "p95_ms": round(pct(0.95), 2), "p99_ms": round(pct(0.99), 2),
                      "max_ms": round(float(lat[-1]) * 1e3, 2), "batches": len(sizes),
                      "mean_batch": round(float(np.mean(sizes)), 1)}))


if __name__ == "__main__":
    main()
