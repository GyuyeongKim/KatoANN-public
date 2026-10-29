// End-to-end checks on synthetic clustered data: recall against brute force,
// identical answers from AMX and AVX-512, and the ranking pass reducing to
// the full scan when 4,000 of the 40,000 candidates survive.
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "katoann/index.h"

using namespace katoann;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

struct Result {
  std::vector<uint32_t> ids;
  std::vector<float> dists;
};

Result run(const Index &idx, const std::vector<float> &q, uint32_t nq, SearchParams p) {
  Result r{std::vector<uint32_t>(static_cast<size_t>(nq) * p.k),
           std::vector<float>(static_cast<size_t>(nq) * p.k)};
  idx.search(q.data(), nq, p, r.ids.data(), r.dists.data());
  return r;
}

// Same distances at every rank; ids may differ only between equal distances.
bool same(const Result &a, const Result &b, uint32_t nq, uint32_t k) {
  for (size_t i = 0; i < static_cast<size_t>(nq) * k; ++i)
    if (a.dists[i] != b.dists[i]) return false;
  for (uint32_t q = 0; q < nq; ++q) {
    const float last = a.dists[q * k + k - 1];
    for (uint32_t j = 0; j < k; ++j)
      if (a.dists[q * k + j] < last && a.ids[q * k + j] != b.ids[q * k + j]) return false;
  }
  return true;
}

double recall(const Result &r, const std::vector<uint32_t> &gt, uint32_t nq, uint32_t k) {
  double hit = 0;
  for (uint32_t q = 0; q < nq; ++q) {
    std::set<uint32_t> t(gt.begin() + q * k, gt.begin() + (q + 1) * k);
    for (uint32_t j = 0; j < k; ++j) hit += t.count(r.ids[q * k + j]);
  }
  return hit / (static_cast<double>(nq) * k);
}

}  // namespace

int main() {
  const uint32_t n = 40000, d = 256, nq = 300, k = 10, nlist = 64;
  std::mt19937 rng(11);
  std::normal_distribution<float> g(0.0f, 1.0f);
  std::vector<float> centers(64 * d);
  for (auto &v : centers) v = g(rng) * 3.0f;
  std::uniform_int_distribution<int> pick(0, 63);
  auto sample = [&](float *out) {
    const float *c = centers.data() + pick(rng) * d;
    double s = 0;
    // Variance decays along the dimensions, as in text embeddings.
    for (uint32_t j = 0; j < d; ++j) {
      out[j] = (c[j] + 0.5f * g(rng)) * std::exp(-static_cast<float>(j) / 40.0f);
      s += out[j] * out[j];
    }
    for (uint32_t j = 0; j < d; ++j) out[j] = static_cast<float>(out[j] / std::sqrt(s));
  };
  std::vector<float> base(static_cast<size_t>(n) * d), q(static_cast<size_t>(nq) * d);
  for (uint32_t i = 0; i < n; ++i) sample(base.data() + static_cast<size_t>(i) * d);
  for (uint32_t i = 0; i < nq; ++i) sample(q.data() + static_cast<size_t>(i) * d);

  std::vector<uint32_t> gt(static_cast<size_t>(nq) * k);
#pragma omp parallel for
  for (int64_t i = 0; i < nq; ++i) {
    std::vector<std::pair<float, uint32_t>> dd(n);
    for (uint32_t j = 0; j < n; ++j) {
      float s = 0;
      for (uint32_t t = 0; t < d; ++t) {
        const float x = q[i * d + t] - base[static_cast<size_t>(j) * d + t];
        s += x * x;
      }
      dd[j] = {s, j};
    }
    std::partial_sort(dd.begin(), dd.begin() + k, dd.end());
    for (uint32_t j = 0; j < k; ++j) gt[i * k + j] = dd[j].second;
  }

  char dir[] = "/tmp/katoann_test_XXXXXX";
  if (!mkdtemp(dir)) return 2;
  const std::string prefix = std::string(dir) + "/idx";
  BuildOptions bo;
  bo.nlist = nlist;
  bo.seed = 1;
  bo.normalized = true;
  bo.rotation_sample = 20000;
  Index::build(base.data(), n, d, bo, prefix);

  const bool amx = amx_available();
  LoadOptions lo;
  lo.use_amx = false;
  Index avx;
  avx.load(prefix, lo);
  check(avx.rotated() && avx.prefix_dims() == 64, "rotation and prefix plane present");

  SearchParams full;
  full.k = k;
  full.nprobe = nlist;
  full.ranking = Ranking::kFull;
  const Result rf = run(avx, q, nq, full);
  check(recall(rf, gt, nq, k) >= 0.99, "full scan of every list matches brute force (recall " +
                                           std::to_string(recall(rf, gt, nq, k)) + ")");

  SearchParams wide = full;
  wide.survivors = 4000;
  for (Ranking r : {Ranking::kHighBytes, Ranking::kPrefix}) {
    wide.ranking = r;
    check(same(rf, run(avx, q, nq, wide), nq, k),
          std::string(r == Ranking::kPrefix ? "prefix" : "high-byte") +
              " ranking with 4,000 of 40,000 candidates surviving equals the full scan");
  }

  SearchParams dflt;
  dflt.k = k;
  dflt.nprobe = 16;
  const Result rt = run(avx, q, nq, dflt);
  SearchParams local = dflt;
  local.selection = Selection::kLocal;
  check(same(rt, run(avx, q, nq, local), nq, k), "local selection equals threshold selection");
  SearchParams full16 = dflt;
  full16.ranking = Ranking::kFull;
  const double r_full = recall(run(avx, q, nq, full16), gt, nq, k);
  const double r_prefix = recall(rt, gt, nq, k);
  check(r_full >= 0.9 && r_prefix >= r_full - 0.02,
        "prefix ranking at nprobe 16 keeps the recall of the full scan (" +
            std::to_string(r_prefix) + " vs " + std::to_string(r_full) + ")");

  LoadOptions flat = lo;
  flat.batched_coarse = false;
  Index avx_flat;
  avx_flat.load(prefix, flat);
  check(recall(run(avx_flat, q, nq, dflt), gt, nq, k) >= r_prefix - 0.02,
        "per-query coarse search");

  if (amx) {
    Index amx_idx;
    amx_idx.load(prefix, LoadOptions{});
    check(amx_idx.engine() == Engine::kAmx, "AMX engine selected");
    for (Ranking r : {Ranking::kFull, Ranking::kHighBytes, Ranking::kPrefix}) {
      SearchParams p = dflt;
      p.ranking = r;
      check(same(run(avx, q, nq, p), run(amx_idx, q, nq, p), nq, k),
            "AMX equals AVX-512, ranking " + std::to_string(static_cast<int>(r)));
    }
  } else {
    std::printf("skip AMX checks (AMX not available)\n");
  }

  LoadOptions bf = lo;
  bf.precision = Precision::kBf16;
  Index bidx;
  bidx.load(prefix, bf);
  check(recall(run(bidx, q, nq, full), gt, nq, k) >= 0.9, "bf16 full scan");

  std::string cmd = std::string("rm -rf ") + dir;
  if (std::system(cmd.c_str()) != 0) return 2;
  std::printf(failures ? "%d failures\n" : "all index checks passed\n", failures);
  return failures ? 1 : 0;
}
