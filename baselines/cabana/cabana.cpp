// CABANA baseline: batched IVF search with bfloat16 matrix products on AMX
// through oneMKL (cblas_gemm_bf16bf16f32), reimplemented from its paper.
// Queries are grouped by list; each list is one product with the queries that
// probe it, and each query keeps its k best of
// every list before a final merge. It reads the list layout of a KatoANN index,
// so both systems scan the same partition, and ranks by inner product on
// unit-length vectors (cosine), like the original.
//
// usage: cabana --index PREFIX --queries Q.fbin [--gt GT.bin] [--k 10]
//               [--nprobe 128,256] [--batch 4096] [--repeat 3] [--threads T]
//               [--query-ids FILE] [--out PREFIX]
//        cabana --index PREFIX --queries Q.fbin --nprobe N --rate QPS
//               [--requests 10000] [--max-batch 4096] [--seed 42] [--threads T]
// The second form serves Poisson arrivals like `katoann load`.
#include <mkl.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <queue>
#include <random>
#include <thread>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint16_t f2bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}

struct Lists {
  int n = 0, d = 0, nlist = 0;
  std::vector<float> centroids;
  std::vector<uint16_t> centroids_bf16;
  std::vector<int64_t> offsets, ids;
  std::vector<uint16_t> vecs;  // row-major bfloat16, list after list
};

Lists load_lists(const std::string &prefix) {
  Lists L;
  uint32_t meta[3];
  {
    std::ifstream f(prefix + ".meta", std::ios::binary);
    f.read(reinterpret_cast<char *>(meta), sizeof(meta));
    if (!f) throw std::runtime_error("cannot read " + prefix + ".meta");
  }
  L.d = meta[0];
  L.nlist = meta[1];
  L.n = meta[2];
  L.centroids.resize(static_cast<size_t>(L.nlist) * L.d);
  {
    std::ifstream f(prefix + ".centroids", std::ios::binary);
    f.read(reinterpret_cast<char *>(L.centroids.data()), L.centroids.size() * 4);
  }
  L.centroids_bf16.resize(L.centroids.size());
  for (size_t i = 0; i < L.centroids.size(); ++i) L.centroids_bf16[i] = f2bf16(L.centroids[i]);
  L.offsets.assign(L.nlist + 1, 0);
  L.ids.resize(L.n);
  L.vecs.resize(static_cast<size_t>(L.n) * L.d);
  std::ifstream f(prefix + ".invlist", std::ios::binary);
  std::vector<uint32_t> ids;
  std::vector<float> v;
  int64_t pos = 0;
  for (int c = 0; c < L.nlist; ++c) {
    uint32_t sz = 0;
    f.read(reinterpret_cast<char *>(&sz), 4);
    ids.resize(sz);
    v.resize(static_cast<size_t>(sz) * L.d);
    f.read(reinterpret_cast<char *>(ids.data()), sz * 4);
    f.read(reinterpret_cast<char *>(v.data()), v.size() * 4);
    for (uint32_t i = 0; i < sz; ++i) L.ids[pos + i] = ids[i];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i)
      L.vecs[static_cast<size_t>(pos) * L.d + i] = f2bf16(v[i]);
    pos += sz;
    L.offsets[c + 1] = pos;
  }
  if (!f || pos != L.n) throw std::runtime_error("truncated " + prefix + ".invlist");
  return L;
}

// Nearest nprobe lists by inner product with the bfloat16 centroids.
void coarse_search(const Lists &L, const uint16_t *q, int nq, int nprobe, std::vector<int> &out) {
  const int d = L.d, nlist = L.nlist;
  std::vector<float> D(static_cast<size_t>(nq) * nlist);
  cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, nq, nlist, d, 1.0f,
                         reinterpret_cast<const MKL_BF16 *>(q), d,
                         reinterpret_cast<const MKL_BF16 *>(L.centroids_bf16.data()), d, 0.0f,
                         D.data(), nlist);
  out.assign(static_cast<size_t>(nq) * nprobe, -1);
#pragma omp parallel
  {
    std::vector<std::pair<float, int>> scored(nlist);
#pragma omp for schedule(static)
    for (int i = 0; i < nq; ++i) {
      const float *di = D.data() + static_cast<size_t>(i) * nlist;
      for (int c = 0; c < nlist; ++c) scored[c] = {-di[c], c};
      const int kp = std::min(nprobe, nlist);
      std::partial_sort(scored.begin(), scored.begin() + kp, scored.end());
      for (int p = 0; p < kp; ++p) out[static_cast<size_t>(i) * nprobe + p] = scored[p].second;
    }
  }
}

void fine_search(const Lists &L, const uint16_t *q, int nq, int nprobe, int K,
                 const std::vector<int> &coarse, int32_t *out_ids) {
  const int d = L.d;
  std::vector<int64_t> pair_c, pair_q;
  pair_c.reserve(static_cast<size_t>(nq) * nprobe);
  pair_q.reserve(static_cast<size_t>(nq) * nprobe);
  for (int i = 0; i < nq; ++i)
    for (int p = 0; p < nprobe; ++p) {
      const int c = coarse[static_cast<size_t>(i) * nprobe + p];
      if (c >= 0) { pair_c.push_back(c); pair_q.push_back(i); }
    }
  const size_t npair = pair_c.size();
  std::vector<size_t> ord(npair);
  {
    std::vector<int64_t> cnt(static_cast<size_t>(L.nlist) + 1, 0);
    for (size_t i = 0; i < npair; ++i) cnt[pair_c[i] + 1]++;
    for (int c = 0; c < L.nlist; ++c) cnt[c + 1] += cnt[c];
    for (size_t i = 0; i < npair; ++i) ord[cnt[pair_c[i]]++] = i;
  }
  struct Block { int list; size_t start; int m; };
  std::vector<Block> blocks;
  for (size_t i = 0; i < npair;) {
    const int64_t c = pair_c[ord[i]];
    size_t j = i;
    while (j < npair && pair_c[ord[j]] == c) j++;
    blocks.push_back({static_cast<int>(c), i, static_cast<int>(j - i)});
    i = j;
  }
  std::vector<int> blk_k(blocks.size());
  std::vector<size_t> blk_off(blocks.size() + 1, 0);
  for (size_t b = 0; b < blocks.size(); ++b) {
    const int nv = static_cast<int>(L.offsets[blocks[b].list + 1] - L.offsets[blocks[b].list]);
    blk_k[b] = std::min(K, nv);
    blk_off[b + 1] = blk_off[b] + static_cast<size_t>(blocks[b].m) * std::max(blk_k[b], 0);
  }
  const size_t total = blk_off.back();
  std::vector<float> sh_d(total);
  std::vector<int32_t> sh_id(total), sh_q(total);
#pragma omp parallel
  {
    mkl_set_num_threads_local(1);
    std::vector<float> ip;
    std::vector<uint16_t> qpack;
    std::vector<std::pair<float, int>> tmp;
#pragma omp for schedule(dynamic, 8)
    for (size_t b = 0; b < blocks.size(); ++b) {
      const int c = blocks[b].list, m = blocks[b].m;
      const size_t s = blocks[b].start;
      const int64_t voff = L.offsets[c];
      const int nv = static_cast<int>(L.offsets[c + 1] - voff);
      if (nv <= 0) continue;
      qpack.resize(static_cast<size_t>(m) * d);
      for (int r = 0; r < m; ++r)
        std::memcpy(qpack.data() + static_cast<size_t>(r) * d,
                    q + static_cast<size_t>(pair_q[ord[s + r]]) * d, static_cast<size_t>(d) * 2);
      ip.assign(static_cast<size_t>(m) * nv, 0.0f);
      const uint16_t *x = L.vecs.data() + static_cast<size_t>(voff) * d;
      cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, m, nv, d, 1.0f,
                             reinterpret_cast<const MKL_BF16 *>(qpack.data()), d,
                             reinterpret_cast<const MKL_BF16 *>(x), d, 0.0f, ip.data(), nv);
      const int kc = blk_k[b];
      for (int r = 0; r < m; ++r) {
        const float *row = ip.data() + static_cast<size_t>(r) * nv;
        tmp.resize(nv);
        for (int t = 0; t < nv; ++t) tmp[t] = {-row[t], t};
        if (kc < nv) std::partial_sort(tmp.begin(), tmp.begin() + kc, tmp.end());
        else std::sort(tmp.begin(), tmp.end());
        const size_t base = blk_off[b] + static_cast<size_t>(r) * kc;
        for (int t = 0; t < kc; ++t) {
          sh_d[base + t] = tmp[t].first;
          sh_id[base + t] = static_cast<int32_t>(L.ids[voff + tmp[t].second]);
          sh_q[base + t] = static_cast<int32_t>(pair_q[ord[s + r]]);
        }
      }
    }
    mkl_set_num_threads_local(0);
  }
  std::vector<int64_t> qcount(nq + 1, 0), qpos(nq, 0);
  for (size_t i = 0; i < total; ++i) qcount[sh_q[i] + 1]++;
  for (int i = 0; i < nq; ++i) qcount[i + 1] += qcount[i];
  std::vector<int32_t> by_q(total);
  for (size_t i = 0; i < total; ++i) by_q[qcount[sh_q[i]] + qpos[sh_q[i]]++] = static_cast<int32_t>(i);
  std::fill(out_ids, out_ids + static_cast<size_t>(nq) * K, -1);
#pragma omp parallel for schedule(dynamic, 64)
  for (int qi = 0; qi < nq; ++qi) {
    std::priority_queue<std::pair<float, int32_t>> heap;
    for (int64_t t = qcount[qi]; t < qcount[qi + 1]; ++t) {
      const int32_t si = by_q[t];
      if (static_cast<int>(heap.size()) < K) heap.push({sh_d[si], sh_id[si]});
      else if (sh_d[si] < heap.top().first) { heap.pop(); heap.push({sh_d[si], sh_id[si]}); }
    }
    for (int t = static_cast<int>(heap.size()) - 1; t >= 0; --t) {
      out_ids[static_cast<size_t>(qi) * K + t] = heap.top().second;
      heap.pop();
    }
  }
}

std::map<std::string, std::string> parse(int argc, char **argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument " + k);
    a[k.substr(2)] = argv[i + 1];
  }
  return a;
}

std::string get(const std::map<std::string, std::string> &a, const std::string &k,
                const std::string &def) {
  auto it = a.find(k);
  return it == a.end() ? def : it->second;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const auto a = parse(argc, argv);
    const int K = std::stoi(get(a, "k", "10")), batch = std::stoi(get(a, "batch", "4096"));
    const int repeat = std::stoi(get(a, "repeat", "3"));
    const int threads = std::stoi(get(a, "threads", std::to_string(omp_get_max_threads())));
    mkl_set_num_threads(threads);
    mkl_set_dynamic(0);
    omp_set_num_threads(threads);
    omp_set_dynamic(0);
    omp_set_max_active_levels(2);
    auto t0 = Clock::now();
    const Lists L = load_lists(get(a, "index", ""));
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::ifstream qf(get(a, "queries", ""), std::ios::binary);
    uint32_t nq = 0, d = 0;
    qf.read(reinterpret_cast<char *>(&nq), 4);
    qf.read(reinterpret_cast<char *>(&d), 4);
    std::vector<float> all(static_cast<size_t>(nq) * d);
    qf.read(reinterpret_cast<char *>(all.data()), all.size() * 4);
    if (!qf || static_cast<int>(d) != L.d) throw std::runtime_error("bad queries");
    std::vector<uint32_t> qid;
    if (a.count("query-ids")) {
      std::ifstream f(a.at("query-ids"));
      uint32_t id;
      while (f >> id) qid.push_back(id);
    } else {
      for (uint32_t i = 0; i < nq; ++i) qid.push_back(i);
    }
    const size_t total = qid.size();
    std::vector<uint16_t> q(total * d);
    for (size_t i = 0; i < total; ++i)
      for (uint32_t j = 0; j < d; ++j) q[i * d + j] = f2bf16(all[static_cast<size_t>(qid[i]) * d + j]);
    std::vector<uint32_t> gt;
    uint32_t gk = 0;
    if (a.count("gt")) {
      std::ifstream f(a.at("gt"), std::ios::binary);
      uint32_t gn = 0;
      f.read(reinterpret_cast<char *>(&gn), 4);
      f.read(reinterpret_cast<char *>(&gk), 4);
      gt.resize(static_cast<size_t>(gn) * gk);
      f.read(reinterpret_cast<char *>(gt.data()), gt.size() * 4);
    }
    MKLVersion v;
    mkl_get_version(&v);
    std::printf("{\"system\": \"cabana\", \"mkl\": \"%d.%d.%d\", \"threads\": %d, \"load_s\": %.1f, "
                "\"queries\": %zu}\n", v.MajorVersion, v.MinorVersion, v.UpdateVersion, threads,
                load_s, total);
    if (a.count("rate")) {
      const double rate = std::stod(a.at("rate"));
      const int n = std::stoi(get(a, "requests", "10000"));
      const int max_batch = std::stoi(get(a, "max-batch", "4096"));
      const int np = std::stoi(get(a, "nprobe", "128"));
      std::mt19937_64 rng(std::stoull(get(a, "seed", "42")));
      std::exponential_distribution<double> gap(rate);
      std::vector<double> arrival(n), latency(n);
      double t = 0;
      for (int i = 0; i < n; ++i) { t += gap(rng); arrival[i] = t; }
      std::vector<uint16_t> bq(static_cast<size_t>(max_batch) * d);
      std::vector<int32_t> bids(static_cast<size_t>(max_batch) * K);
      std::vector<int> coarse;
      const int warm = static_cast<int>(std::min<size_t>(max_batch, total));
      coarse_search(L, q.data(), warm, np, coarse);
      fine_search(L, q.data(), warm, np, K, coarse, bids.data());
      const auto start = Clock::now();
      auto now = [&]() { return std::chrono::duration<double>(Clock::now() - start).count(); };
      int next = 0;
      size_t batches = 0;
      while (next < n) {
        double tn = now();
        if (arrival[next] > tn) {
          std::this_thread::sleep_for(std::chrono::duration<double>(arrival[next] - tn));
          tn = now();
        }
        int end = next;
        while (end < n && end - next < max_batch && arrival[end] <= tn) ++end;
        const int b = end - next;
        for (int i = 0; i < b; ++i)
          std::memcpy(bq.data() + static_cast<size_t>(i) * d,
                      q.data() + static_cast<size_t>((next + i) % total) * d, static_cast<size_t>(d) * 2);
        coarse_search(L, bq.data(), b, np, coarse);
        fine_search(L, bq.data(), b, np, K, coarse, bids.data());
        const double done = now();
        for (int i = 0; i < b; ++i) latency[next + i] = done - arrival[next + i];
        ++batches;
        next = end;
      }
      const double span = now();
      std::sort(latency.begin(), latency.end());
      auto pct = [&](double x) {
        return latency[std::min<size_t>(latency.size() - 1, static_cast<size_t>(std::ceil(x * n)) - 1)];
      };
      std::printf("{\"system\": \"cabana\", \"rate\": %.1f, \"requests\": %d, \"nprobe\": %d, "
                  "\"achieved_qps\": %.1f, \"p50_ms\": %.2f, \"p95_ms\": %.2f, \"p99_ms\": %.2f, "
                  "\"max_ms\": %.2f, \"batches\": %zu, \"mean_batch\": %.1f}\n",
                  rate, n, np, n / span, pct(0.5) * 1e3, pct(0.95) * 1e3, pct(0.99) * 1e3,
                  latency.back() * 1e3, batches, static_cast<double>(n) / batches);
      return 0;
    }
    std::vector<int32_t> ids(total * K);
    std::stringstream ss(get(a, "nprobe", "128"));
    std::string item;
    while (std::getline(ss, item, ',')) {
      const int np = std::stoi(item);
      auto pass = [&]() {
        std::vector<int> coarse;
        for (size_t off = 0; off < total; off += batch) {
          const int b = static_cast<int>(std::min<size_t>(batch, total - off));
          coarse_search(L, q.data() + off * d, b, np, coarse);
          fine_search(L, q.data() + off * d, b, np, K, coarse, ids.data() + off * K);
        }
      };
      pass();
      std::vector<double> secs;
      for (int r = 0; r < repeat; ++r) {
        t0 = Clock::now();
        pass();
        secs.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
      }
      std::sort(secs.begin(), secs.end());
      const double med = secs[secs.size() / 2];
      std::printf("{\"nprobe\": %d, \"k\": %d, \"batch\": %d, \"qps\": %.1f, \"seconds_median\": %.4f",
                  np, K, batch, total / med, med);
      if (!gt.empty()) {
        double hit = 0;
        for (size_t i = 0; i < total; ++i) {
          std::set<uint32_t> t(gt.begin() + static_cast<size_t>(qid[i]) * gk,
                               gt.begin() + static_cast<size_t>(qid[i]) * gk + K);
          for (int j = 0; j < K; ++j) hit += ids[i * K + j] >= 0 && t.count(ids[i * K + j]);
        }
        std::printf(", \"recall\": %.5f", hit / (static_cast<double>(total) * K));
      }
      std::printf("}\n");
      std::fflush(stdout);
      if (a.count("out")) {
        std::ofstream f(a.at("out") + ".np" + std::to_string(np) + ".bin", std::ios::binary);
        const uint32_t hdr[2] = {static_cast<uint32_t>(total), static_cast<uint32_t>(K)};
        f.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
        f.write(reinterpret_cast<const char *>(ids.data()), ids.size() * 4);
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "cabana: %s\n", e.what());
    return 1;
  }
}
