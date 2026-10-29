#include "katoann/index.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>

#include "kmeans.h"

namespace katoann {

namespace {

constexpr uint32_t kBlock = 256;  // database vectors per scan block
constexpr uint32_t kDimcMagic = 0x434D4944u;  // "DIMC"
constexpr uint32_t kDimcVersion = 1u;

using PFI = std::pair<float, uint32_t>;

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

inline float l2_fp32(const float *a, const float *b, uint32_t d) {
#if defined(__AVX512F__)
  __m512 acc = _mm512_setzero_ps();
  uint32_t i = 0;
  for (; i + 16 <= d; i += 16) {
    __m512 dd = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    acc = _mm512_fmadd_ps(dd, dd, acc);
  }
  float tail = 0;
  for (; i < d; ++i) {
    float diff = a[i] - b[i];
    tail += diff * diff;
  }
  return _mm512_reduce_add_ps(acc) + tail;
#else
  float s = 0;
  for (uint32_t i = 0; i < d; ++i) {
    float diff = a[i] - b[i];
    s += diff * diff;
  }
  return s;
#endif
}

// Keeps the K smallest (distance, id) pairs of a row in a max-heap. Sixteen or
// thirty-two distances are compared with the current K-th distance at once,
// and only those below it reach the heap.
inline void reduce_row_topk(std::priority_queue<PFI> &h, const float *row,
                            const uint32_t *ids, uint32_t blk, uint32_t K) {
  auto worst = [&]() -> float {
    return (h.size() >= K) ? h.top().first : std::numeric_limits<float>::max();
  };
  auto push1 = [&](uint32_t idx) {
    const float dd = row[idx];
    if (h.size() < K) h.emplace(dd, ids[idx]);
    else if (dd < h.top().first) { h.pop(); h.emplace(dd, ids[idx]); }
  };
#if defined(__AVX512F__)
  float thr = worst();
  uint32_t i = 0;
  for (; i + 32 <= blk; i += 32) {
    const __m512 t = _mm512_set1_ps(thr);
    __mmask16 m0 = _mm512_cmp_ps_mask(_mm512_loadu_ps(row + i), t, _CMP_LT_OS);
    __mmask16 m1 = _mm512_cmp_ps_mask(_mm512_loadu_ps(row + i + 16), t, _CMP_LT_OS);
    if (!(m0 | m1)) continue;
    while (m0) { uint32_t b = __builtin_ctz(m0); m0 &= m0 - 1; push1(i + b); }
    while (m1) { uint32_t b = __builtin_ctz(m1); m1 &= m1 - 1; push1(i + 16 + b); }
    thr = worst();
  }
  for (; i + 16 <= blk; i += 16) {
    __mmask16 mk = _mm512_cmp_ps_mask(_mm512_loadu_ps(row + i),
                                      _mm512_set1_ps(thr), _CMP_LT_OS);
    if (!mk) continue;
    while (mk) { uint32_t b = __builtin_ctz(mk); mk &= mk - 1; push1(i + b); }
    thr = worst();
  }
  for (; i < blk; ++i) push1(i);
#else
  (void) worst;
  for (uint32_t i = 0; i < blk; ++i) push1(i);
#endif
}

// Keeps the S smallest pairs of one (query, list) in ascending order in kr/ir.
// Ties keep scan order.
inline void select_row_topS_sorted(float *kr, uint32_t *ir, const float *row,
                                   const uint32_t *ids, uint32_t blk, uint32_t S) {
  auto insert1 = [&](uint32_t idx) {
    const float dd = row[idx];
    if (dd >= kr[S - 1]) return;
    uint32_t i = S - 1;
    while (i > 0 && kr[i - 1] > dd) {
      kr[i] = kr[i - 1];
      ir[i] = ir[i - 1];
      --i;
    }
    kr[i] = dd;
    ir[i] = ids[idx];
  };
#if defined(__AVX512F__)
  uint32_t i = 0;
  for (; i + 32 <= blk; i += 32) {
    const __m512 t = _mm512_set1_ps(kr[S - 1]);
    __mmask16 m0 = _mm512_cmp_ps_mask(_mm512_loadu_ps(row + i), t, _CMP_LT_OS);
    __mmask16 m1 = _mm512_cmp_ps_mask(_mm512_loadu_ps(row + i + 16), t, _CMP_LT_OS);
    if (!(m0 | m1)) continue;
    while (m0) { uint32_t b = __builtin_ctz(m0); m0 &= m0 - 1; insert1(i + b); }
    while (m1) { uint32_t b = __builtin_ctz(m1); m1 &= m1 - 1; insert1(i + 16 + b); }
  }
  for (; i + 16 <= blk; i += 16) {
    __mmask16 mk = _mm512_cmp_ps_mask(_mm512_loadu_ps(row + i),
                                      _mm512_set1_ps(kr[S - 1]), _CMP_LT_OS);
    if (!mk) continue;
    while (mk) { uint32_t b = __builtin_ctz(mk); mk &= mk - 1; insert1(i + b); }
  }
  for (; i < blk; ++i) insert1(i);
#else
  for (uint32_t i = 0; i < blk; ++i) insert1(i);
#endif
}

// out[i][j] = <in_i, rot_j>. Basis rows are the outer loop so the rotation
// streams once per row block.
void rotate_rows_serial(const float *in, float *out, size_t nrows, uint32_t d,
                        const float *rot) {
  uint32_t j = 0;
#if defined(__AVX512F__)
  for (; j + 4 <= d; j += 4) {
    const float *r0 = rot + static_cast<size_t>(j) * d;
    const float *r1 = r0 + d;
    const float *r2 = r1 + d;
    const float *r3 = r2 + d;
    for (size_t i = 0; i < nrows; ++i) {
      const float *x = in + i * d;
      float *y = out + i * d;
      __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
      __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
      uint32_t k = 0;
      for (; k + 16 <= d; k += 16) {
        const __m512 xv = _mm512_loadu_ps(x + k);
        a0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r0 + k), a0);
        a1 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r1 + k), a1);
        a2 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r2 + k), a2);
        a3 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(r3 + k), a3);
      }
      float s0 = _mm512_reduce_add_ps(a0), s1 = _mm512_reduce_add_ps(a1);
      float s2 = _mm512_reduce_add_ps(a2), s3 = _mm512_reduce_add_ps(a3);
      for (; k < d; ++k) {
        s0 += x[k] * r0[k]; s1 += x[k] * r1[k];
        s2 += x[k] * r2[k]; s3 += x[k] * r3[k];
      }
      y[j] = s0; y[j + 1] = s1; y[j + 2] = s2; y[j + 3] = s3;
    }
  }
#endif
  for (; j < d; ++j) {
    const float *r = rot + static_cast<size_t>(j) * d;
    for (size_t i = 0; i < nrows; ++i) {
      const float *x = in + i * d;
      float s = 0.0f;
      for (uint32_t k = 0; k < d; ++k) s += x[k] * r[k];
      out[i * d + j] = s;
    }
  }
}

void rotate_rows(const float *in, float *out, size_t nrows, uint32_t d,
                 const float *rot) {
  constexpr size_t RB = 128;
  const size_t nblk = (nrows + RB - 1) / RB;
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t b = 0; b < static_cast<int64_t>(nblk); ++b) {
    const size_t r0 = static_cast<size_t>(b) * RB;
    const size_t rn = std::min(RB, nrows - r0);
    rotate_rows_serial(in + r0 * d, out + r0 * d, rn, d, rot);
  }
}

// Principal axes of a sample (n x d, consumed) by cyclic Jacobi on the
// covariance. Row j of rot_out is the axis with the j-th largest variance.
// Jacobi applies exact Givens rotations, so the basis is orthogonal whether
// or not the sweeps converge.
bool pca_rotation(std::vector<float> &sample, size_t n, uint32_t d,
                  uint32_t max_sweeps, std::vector<float> &rot_out) {
  if (n < 2 || d < 2 || sample.size() < n * static_cast<size_t>(d)) return false;
  const size_t dd = static_cast<size_t>(d) * d;
  {
    std::vector<double> mean(d, 0.0);
    for (size_t s = 0; s < n; ++s) {
      const float *x = sample.data() + s * d;
      for (uint32_t k = 0; k < d; ++k) mean[k] += x[k];
    }
    for (uint32_t k = 0; k < d; ++k) mean[k] /= static_cast<double>(n);
#pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < static_cast<int64_t>(n); ++s) {
      float *x = sample.data() + static_cast<size_t>(s) * d;
      for (uint32_t k = 0; k < d; ++k) x[k] = static_cast<float>(x[k] - mean[k]);
    }
  }
  std::vector<double> A(dd, 0.0);
  {
    constexpr size_t CH = 256;
    for (size_t s0 = 0; s0 < n; s0 += CH) {
      const size_t sn = std::min(CH, n - s0);
#pragma omp parallel for schedule(dynamic, 8)
      for (int64_t i = 0; i < static_cast<int64_t>(d); ++i) {
        double *ci = A.data() + static_cast<size_t>(i) * d;
        for (size_t s = 0; s < sn; ++s) {
          const float *x = sample.data() + (s0 + s) * d;
          const double xi = x[i];
          for (uint32_t j = static_cast<uint32_t>(i); j < d; ++j) ci[j] += xi * x[j];
        }
      }
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    for (uint32_t i = 0; i < d; ++i)
      for (uint32_t j = i; j < d; ++j) {
        const double v = A[static_cast<size_t>(i) * d + j] * inv_n;
        A[static_cast<size_t>(i) * d + j] = v;
        A[static_cast<size_t>(j) * d + i] = v;
      }
  }
  max_sweeps = std::clamp<uint32_t>(max_sweeps, 1u, 64u);
  std::vector<double> V(dd, 0.0);
  for (uint32_t i = 0; i < d; ++i) V[static_cast<size_t>(i) * d + i] = 1.0;
  double diag_norm = 0.0;
  for (uint32_t i = 0; i < d; ++i) {
    const double a = A[static_cast<size_t>(i) * d + i];
    diag_norm += a * a;
  }
  const double stop_off = 1e-18 * std::max(diag_norm, 1e-300);
  for (uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
    double off = 0.0;
    for (uint32_t i = 0; i < d; ++i)
      for (uint32_t j = i + 1; j < d; ++j) {
        const double a = A[static_cast<size_t>(i) * d + j];
        off += 2.0 * a * a;
      }
    std::fprintf(stderr, "rotation: Jacobi sweep %u/%u, off-diagonal %.3e\n",
                 sweep + 1, max_sweeps, off);
    if (off <= stop_off) break;
    for (uint32_t p = 0; p + 1 < d; ++p) {
      for (uint32_t q = p + 1; q < d; ++q) {
        const double apq = A[static_cast<size_t>(p) * d + q];
        const double app = A[static_cast<size_t>(p) * d + p];
        const double aqq = A[static_cast<size_t>(q) * d + q];
        if (std::fabs(apq) <= 1e-14 * std::sqrt(std::fabs(app * aqq)) + 1e-300) continue;
        const double theta = 0.5 * (aqq - app) / apq;
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (std::fabs(theta) + std::sqrt(1.0 + theta * theta));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        double *rp = A.data() + static_cast<size_t>(p) * d;
        double *rq = A.data() + static_cast<size_t>(q) * d;
        for (uint32_t k = 0; k < d; ++k) {
          const double xp = rp[k], xq = rq[k];
          rp[k] = c * xp - s * xq;
          rq[k] = s * xp + c * xq;
        }
        for (uint32_t k = 0; k < d; ++k) {
          double *row = A.data() + static_cast<size_t>(k) * d;
          const double xp = row[p], xq = row[q];
          row[p] = c * xp - s * xq;
          row[q] = s * xp + c * xq;
        }
        for (uint32_t k = 0; k < d; ++k) {
          double *row = V.data() + static_cast<size_t>(k) * d;
          const double xp = row[p], xq = row[q];
          row[p] = c * xp - s * xq;
          row[q] = s * xp + c * xq;
        }
      }
    }
  }
  std::vector<std::pair<double, uint32_t>> ev(d);
  for (uint32_t i = 0; i < d; ++i) ev[i] = {A[static_cast<size_t>(i) * d + i], i};
  std::sort(ev.begin(), ev.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  rot_out.resize(dd);
  for (uint32_t j = 0; j < d; ++j) {
    const uint32_t src = ev[j].second;
    for (uint32_t k = 0; k < d; ++k)
      rot_out[static_cast<size_t>(j) * d + k] =
          static_cast<float>(V[static_cast<size_t>(k) * d + src]);
  }
  return true;
}

uint32_t round_up(uint32_t x, uint32_t m) { return (x + m - 1) / m * m; }

void emit_topk(std::priority_queue<PFI> &h, uint32_t K, uint32_t *ids, float *dists) {
  std::vector<PFI> tmp;
  tmp.reserve(h.size());
  while (!h.empty()) { tmp.push_back(h.top()); h.pop(); }
  std::reverse(tmp.begin(), tmp.end());
  for (uint32_t i = 0; i < K; ++i) {
    if (i < tmp.size()) {
      dists[i] = tmp[i].first;
      ids[i] = tmp[i].second;
    } else {
      dists[i] = std::numeric_limits<float>::max();
      ids[i] = UINT32_MAX;
    }
  }
}

using WorkList = std::vector<std::pair<uint32_t, std::vector<uint32_t>>>;

// Inverts the probe lists: for every list, the queries that probe it.
WorkList invert(const uint32_t *probed, uint32_t nq, uint32_t nprobe) {
  std::unordered_map<uint32_t, std::vector<uint32_t>> c2q;
  c2q.reserve(static_cast<size_t>(nq) * nprobe / 4 + 16);
  for (uint32_t qi = 0; qi < nq; ++qi) {
    const uint32_t *pr = probed + static_cast<size_t>(qi) * nprobe;
    for (uint32_t p = 0; p < nprobe; ++p) c2q[pr[p]].push_back(qi);
  }
  return WorkList(std::make_move_iterator(c2q.begin()),
                  std::make_move_iterator(c2q.end()));
}

}  // namespace

// ---------------------------------------------------------------------------
// Build

void Index::build(const float *data, uint32_t n, uint32_t dim,
                  const BuildOptions &opt, const std::string &prefix) {
  if (dim == 0 || dim > kMaxDim) throw std::invalid_argument("dimension must be 1 to 4096");
  if (n == 0 || opt.nlist == 0 || opt.nlist > n)
    throw std::invalid_argument("need at least nlist vectors");
  Index idx;
  idx.dim_ = dim;
  idx.nlist_ = opt.nlist;
  idx.npts_ = n;
  idx.normalized_ = opt.normalized;

  const uint32_t ts = std::min(opt.train_size, n);
  if (ts > (1u << 23)) throw std::invalid_argument("train_size is limited to 8388608");
  std::vector<float> train(static_cast<size_t>(ts) * dim);
  if (ts == n) {
    std::memcpy(train.data(), data, train.size() * sizeof(float));
  } else {
    const uint64_t stride = static_cast<uint64_t>(n) * (1ULL << 32) / ts;
    for (uint32_t i = 0; i < ts; ++i) {
      uint32_t src = static_cast<uint32_t>((static_cast<uint64_t>(i) * stride) >> 32);
      if (src >= n) src = n - 1;
      std::memcpy(train.data() + static_cast<size_t>(i) * dim,
                  data + static_cast<size_t>(src) * dim, dim * sizeof(float));
    }
  }
  auto t0 = std::chrono::steady_clock::now();
  idx.centroids_.assign(static_cast<size_t>(opt.nlist) * dim, 0.0f);
  kmeans::kmeanspp(train.data(), ts, dim, idx.centroids_.data(), opt.nlist, opt.seed);
  std::fprintf(stderr, "build: k-means++ seeding of %u lists from %u vectors, %.1f s\n",
               opt.nlist, ts, seconds_since(t0));
  t0 = std::chrono::steady_clock::now();
  kmeans::elkan(train.data(), ts, dim, idx.centroids_.data(), opt.nlist, opt.kmeans_iters);
  std::fprintf(stderr, "build: k-means, %u iterations at most, %.1f s\n",
               opt.kmeans_iters, seconds_since(t0));
  train.clear();
  train.shrink_to_fit();

  t0 = std::chrono::steady_clock::now();
  std::vector<uint32_t> assign(n);
  constexpr uint32_t kAssignBlock = 16384;
  for (uint32_t off = 0; off < n; off += kAssignBlock) {
    const uint32_t cur = std::min(kAssignBlock, n - off);
    kmeans::closest_centers(data + static_cast<size_t>(off) * dim, cur, dim,
                            idx.centroids_.data(), opt.nlist, 1, assign.data() + off);
  }
  std::vector<uint32_t> csize(opt.nlist, 0);
  for (uint32_t i = 0; i < n; ++i) csize[assign[i]]++;
  idx.ids_.assign(opt.nlist, {});
  idx.raw_.assign(opt.nlist, {});
  for (uint32_t c = 0; c < opt.nlist; ++c) {
    idx.ids_[c].reserve(csize[c]);
    idx.raw_[c].reserve(static_cast<size_t>(csize[c]) * dim);
  }
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t c = assign[i];
    idx.ids_[c].push_back(i);
    const float *v = data + static_cast<size_t>(i) * dim;
    idx.raw_[c].insert(idx.raw_[c].end(), v, v + dim);
  }
  std::fprintf(stderr, "build: assigned %u vectors (lists of %u to %u), %.1f s\n", n,
               *std::min_element(csize.begin(), csize.end()),
               *std::max_element(csize.begin(), csize.end()), seconds_since(t0));

  {
    std::ofstream f(prefix + ".meta", std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + prefix + ".meta");
    const uint32_t meta[5] = {dim, opt.nlist, n, 0u, opt.normalized ? 1u : 0u};
    f.write(reinterpret_cast<const char *>(meta), sizeof(meta));
  }
  {
    std::ofstream f(prefix + ".centroids", std::ios::binary);
    f.write(reinterpret_cast<const char *>(idx.centroids_.data()),
            idx.centroids_.size() * sizeof(float));
  }
  {
    std::ofstream f(prefix + ".invlist", std::ios::binary);
    std::ofstream g(prefix + ".norms", std::ios::binary);
    for (uint32_t c = 0; c < opt.nlist; ++c) {
      const uint32_t sz = static_cast<uint32_t>(idx.ids_[c].size());
      f.write(reinterpret_cast<const char *>(&sz), sizeof(sz));
      f.write(reinterpret_cast<const char *>(idx.ids_[c].data()), sz * sizeof(uint32_t));
      f.write(reinterpret_cast<const char *>(idx.raw_[c].data()),
              static_cast<size_t>(sz) * dim * sizeof(float));
      g.write(reinterpret_cast<const char *>(&sz), sizeof(sz));
    }
    if (!f || !g) throw std::runtime_error("cannot write " + prefix + ".invlist");
  }

  t0 = std::chrono::steady_clock::now();
  idx.compute_rotation_(opt.rotation_sample, opt.jacobi_sweeps);
  std::fprintf(stderr, "build: rotation, %.1f s\n", seconds_since(t0));
  idx.rotate_lists_();
  idx.quantize_sq16_();
  idx.build_head_(std::min(std::max(64u, round_up(dim / 4u, 64u)), idx.dc_));
  idx.write_dimc_(prefix);
}

void Index::compute_rotation_(uint32_t sample_rows, uint32_t sweeps) {
  const size_t target = std::min<size_t>(std::max<uint32_t>(sample_rows, 1024u), npts_);
  const size_t stride = std::max<size_t>(1, npts_ / target);
  std::vector<float> sample;
  sample.reserve(target * dim_);
  size_t taken = 0, gi = 0;
  for (uint32_t c = 0; c < nlist_ && taken < target; ++c) {
    const float *vec0 = raw_[c].data();
    const size_t N = ids_[c].size();
    for (size_t i = 0; i < N && taken < target; ++i, ++gi) {
      if (gi % stride) continue;
      sample.insert(sample.end(), vec0 + i * dim_, vec0 + (i + 1) * dim_);
      ++taken;
    }
  }
  rotated_ = taken >= 2 && pca_rotation(sample, taken, dim_, sweeps, rotation_);
  if (!rotated_) rotation_.clear();
}

void Index::write_dimc_(const std::string &prefix) const {
  std::ofstream f(prefix + ".dimc", std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + prefix + ".dimc");
  const uint32_t hdr[8] = {kDimcMagic, kDimcVersion, dim_, nlist_, npts_,
                           head_dims_, rotated_ ? 1u : 0u, 0u};
  f.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
  if (rotated_)
    f.write(reinterpret_cast<const char *>(rotation_.data()),
            static_cast<std::streamsize>(rotation_.size() * sizeof(float)));
  for (uint32_t c = 0; c < nlist_; ++c) {
    const uint32_t N = static_cast<uint32_t>(ids_[c].size());
    f.write(reinterpret_cast<const char *>(&N), sizeof(N));
    if (N == 0) continue;
    f.write(reinterpret_cast<const char *>(head_norm_[c].data()),
            static_cast<std::streamsize>(N * sizeof(int64_t)));
    f.write(reinterpret_cast<const char *>(head_[c].data()),
            static_cast<std::streamsize>(head_[c].size()));
  }
  if (!f) throw std::runtime_error("cannot write " + prefix + ".dimc");
}

// ---------------------------------------------------------------------------
// Load

void Index::read_files_(const std::string &prefix) {
  {
    std::ifstream f(prefix + ".meta", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + prefix + ".meta");
    uint32_t meta[3] = {0, 0, 0};
    f.read(reinterpret_cast<char *>(meta), sizeof(meta));
    if (!f) throw std::runtime_error("truncated " + prefix + ".meta");
    dim_ = meta[0];
    nlist_ = meta[1];
    npts_ = meta[2];
    uint32_t extra[2] = {0, 0};
    f.read(reinterpret_cast<char *>(extra), sizeof(extra));
    if (f.gcount() == static_cast<std::streamsize>(sizeof(extra))) {
      if (extra[0] != 0) throw std::runtime_error("only the L2 metric is supported");
      normalized_ = extra[1] != 0;
    }
  }
  if (dim_ == 0 || dim_ > kMaxDim || nlist_ == 0)
    throw std::runtime_error("bad header in " + prefix + ".meta");
  centroids_.assign(static_cast<size_t>(nlist_) * dim_, 0.0f);
  {
    std::ifstream f(prefix + ".centroids", std::ios::binary);
    f.read(reinterpret_cast<char *>(centroids_.data()), centroids_.size() * sizeof(float));
    if (!f) throw std::runtime_error("cannot read " + prefix + ".centroids");
  }
  ids_.assign(nlist_, {});
  raw_.assign(nlist_, {});
  std::ifstream f(prefix + ".invlist", std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + prefix + ".invlist");
  uint64_t total = 0;
  for (uint32_t c = 0; c < nlist_; ++c) {
    uint32_t sz = 0;
    f.read(reinterpret_cast<char *>(&sz), sizeof(sz));
    ids_[c].resize(sz);
    f.read(reinterpret_cast<char *>(ids_[c].data()), sz * sizeof(uint32_t));
    raw_[c].resize(static_cast<size_t>(sz) * dim_);
    f.read(reinterpret_cast<char *>(raw_[c].data()),
           static_cast<size_t>(sz) * dim_ * sizeof(float));
    total += sz;
  }
  if (!f || total != npts_) throw std::runtime_error("truncated " + prefix + ".invlist");
}

bool Index::read_rotation_(const std::string &prefix, uint32_t *stored_dims) {
  std::ifstream f(prefix + ".dimc", std::ios::binary);
  if (!f) return false;
  uint32_t hdr[8] = {0};
  f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
  if (!f || hdr[0] != kDimcMagic || hdr[1] != kDimcVersion)
    throw std::runtime_error("bad header in " + prefix + ".dimc");
  if (hdr[2] != dim_ || hdr[3] != nlist_ || hdr[4] != npts_)
    throw std::runtime_error(prefix + ".dimc does not match the index");
  *stored_dims = hdr[5];
  rotated_ = hdr[6] == 1u;
  if (rotated_) {
    rotation_.resize(static_cast<size_t>(dim_) * dim_);
    f.read(reinterpret_cast<char *>(rotation_.data()),
           static_cast<std::streamsize>(rotation_.size() * sizeof(float)));
    if (!f) throw std::runtime_error("truncated rotation in " + prefix + ".dimc");
  }
  return true;
}

void Index::rotate_lists_() {
  if (!rotated_) return;
  const uint32_t d = dim_;
  const float *rot = rotation_.data();
#pragma omp parallel
  {
    std::vector<float> tmp;
#pragma omp for schedule(dynamic, 1)
    for (int64_t c = 0; c < static_cast<int64_t>(nlist_); ++c) {
      const size_t N = ids_[c].size();
      if (N == 0 || raw_[c].empty()) continue;
      tmp.resize(N * d);
      rotate_rows_serial(raw_[c].data(), tmp.data(), N, d, rot);
      std::memcpy(raw_[c].data(), tmp.data(), N * static_cast<size_t>(d) * sizeof(float));
    }
  }
}

// One scale for all vectors: v = round((x - lo) * 65535 / (hi - lo)).
void Index::quantize_sq16_() {
  dc_ = round_up(dim_, 64u);
  float glo = std::numeric_limits<float>::max();
  float ghi = -std::numeric_limits<float>::max();
#pragma omp parallel for schedule(dynamic, 8) reduction(min : glo) reduction(max : ghi)
  for (int64_t c = 0; c < static_cast<int64_t>(nlist_); ++c) {
    const float *vec0 = raw_[c].data();
    const size_t cnt = raw_[c].size();
    for (size_t k = 0; k < cnt; ++k) {
      const float v = vec0[k];
      glo = std::min(glo, v);
      ghi = std::max(ghi, v);
    }
  }
  sq_lo_ = glo;
  const double rng = static_cast<double>(ghi) - glo;
  sq_inv_step_ = 65535.0 / rng;
  sq_step_ = rng / 65535.0;
  sq_step_sq_ = sq_step_ * sq_step_;
  hi_.assign(nlist_, {});
  lo_.assign(nlist_, {});
  norm_.assign(nlist_, {});
  hi_norm_.assign(nlist_, {});
  const uint32_t Dc = dc_;
#pragma omp parallel for schedule(dynamic, 8)
  for (int64_t c = 0; c < static_cast<int64_t>(nlist_); ++c) {
    const uint32_t N = static_cast<uint32_t>(ids_[c].size());
    if (N == 0) continue;
    const float *vec0 = raw_[c].data();
    std::vector<uint8_t> hp(static_cast<size_t>(N) * Dc, 0);
    std::vector<uint8_t> lp(static_cast<size_t>(N) * Dc, 0);
    norm_[c].resize(N);
    hi_norm_[c].resize(N);
    for (uint32_t i = 0; i < N; ++i) {
      const float *v = vec0 + static_cast<size_t>(i) * dim_;
      int64_t nrm = 0, nrm_hi = 0;
      for (uint32_t k = 0; k < dim_; ++k) {
        long r = std::lround((static_cast<double>(v[k]) - sq_lo_) * sq_inv_step_);
        if (r < 0) r = 0; else if (r > 65535) r = 65535;
        const uint32_t q = static_cast<uint32_t>(r);
        const uint8_t qh = static_cast<uint8_t>(q >> 8);
        hp[static_cast<size_t>(i) * Dc + k] = qh;
        lp[static_cast<size_t>(i) * Dc + k] = static_cast<uint8_t>(q & 0xFF);
        nrm += static_cast<int64_t>(q) * q;
        nrm_hi += static_cast<int64_t>(qh) * qh;
      }
      norm_[c][i] = nrm;
      hi_norm_[c][i] = nrm_hi;
    }
    const size_t psz = plane_bytes(N, Dc);
    hi_[c].resize(psz);
    lo_[c].resize(psz);
    pack_plane(hp.data(), N, Dc, hi_[c].data());
    pack_plane(lp.data(), N, Dc, lo_[c].data());
    raw_[c].clear();
    raw_[c].shrink_to_fit();
  }
}

// The prefix plane holds the first `dims` high bytes of every vector, copied
// out of the packed high plane and packed again with its own width.
void Index::build_head_(uint32_t dims) {
  if (dims == 0 || dims % 64 || dims > dc_)
    throw std::invalid_argument("prefix_dims must be a multiple of 64 within the dimension");
  const uint32_t nc = dc_ / 64;
  head_.assign(nlist_, {});
  head_norm_.assign(nlist_, {});
#pragma omp parallel for schedule(dynamic, 1)
  for (uint32_t c = 0; c < nlist_; ++c) {
    const uint32_t n = static_cast<uint32_t>(ids_[c].size());
    const size_t bytes = plane_bytes(n, dims);
    std::vector<uint8_t> rows(bytes, 0);
    head_norm_[c].assign(n, 0);
    for (uint32_t i = 0; i < n; ++i) {
      int64_t norm = 0;
      for (uint32_t k = 0; k < dims; ++k) {
        const size_t off = (static_cast<size_t>(i / 16) * nc + k / 64) * 1024 +
                           (k % 64 / 4) * 64 + (i % 16) * 4 + k % 4;
        const uint8_t v = hi_[c][off];
        rows[static_cast<size_t>(i) * dims + k] = v;
        norm += static_cast<int64_t>(v) * v;
      }
      head_norm_[c][i] = norm;
    }
    head_[c].resize(bytes);
    pack_plane(rows.data(), n, dims, head_[c].data());
  }
  head_dims_ = dims;
}

// Centroids quantized with the scale of the database, packed like one list.
void Index::build_coarse_planes_() {
  const uint32_t Dc = dc_;
  const float *cen_src = centroids_.data();
  std::vector<float> cen_rot;
  if (rotated_) {
    cen_rot.resize(static_cast<size_t>(nlist_) * dim_);
    rotate_rows(centroids_.data(), cen_rot.data(), nlist_, dim_, rotation_.data());
    cen_src = cen_rot.data();
  }
  std::vector<uint8_t> chp(static_cast<size_t>(nlist_) * Dc, 0);
  std::vector<uint8_t> clp(static_cast<size_t>(nlist_) * Dc, 0);
  cen_norm_.assign(nlist_, 0);
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t c = 0; c < static_cast<int64_t>(nlist_); ++c) {
    const float *v = cen_src + static_cast<size_t>(c) * dim_;
    int64_t nrm = 0;
    for (uint32_t k = 0; k < dim_; ++k) {
      long r = std::lround((static_cast<double>(v[k]) - sq_lo_) * sq_inv_step_);
      if (r < 0) r = 0; else if (r > 65535) r = 65535;
      const uint32_t q = static_cast<uint32_t>(r);
      chp[static_cast<size_t>(c) * Dc + k] = static_cast<uint8_t>(q >> 8);
      clp[static_cast<size_t>(c) * Dc + k] = static_cast<uint8_t>(q & 0xFF);
      nrm += static_cast<int64_t>(q) * q;
    }
    cen_norm_[c] = nrm;
  }
  const size_t cpsz = plane_bytes(nlist_, Dc);
  cen_hi_.resize(cpsz);
  cen_lo_.resize(cpsz);
  pack_plane(chp.data(), nlist_, Dc, cen_hi_.data());
  pack_plane(clp.data(), nlist_, Dc, cen_lo_.data());
  coarse_batched_ = true;
}

void Index::build_bf16_() {
  const uint32_t dp = round_up(dim_, 32u);
  bf16_.assign(nlist_, {});
  bf16_norm_.assign(nlist_, {});
  for (uint32_t c = 0; c < nlist_; ++c) {
    const uint32_t N = static_cast<uint32_t>(ids_[c].size());
    if (N == 0) continue;
    const float *src = raw_[c].data();
    std::vector<uint16_t> rows(static_cast<size_t>(N) * dp, 0);
    for (uint32_t i = 0; i < N; ++i)
      to_bf16(src + static_cast<size_t>(i) * dim_, rows.data() + static_cast<size_t>(i) * dp, dim_);
    bf16_norm_[c].resize(N);
    for (uint32_t i = 0; i < N; ++i)
      bf16_norm_[c][i] = bf16_norm(rows.data() + static_cast<size_t>(i) * dp, dim_);
    if (engine_ == Engine::kAmx) {
      bf16_[c].resize(bf16_packed_elems(N, dp));
      pack_bf16(rows.data(), N, dp, bf16_[c].data());
    } else {
      bf16_[c] = std::move(rows);
    }
    raw_[c].clear();
    raw_[c].shrink_to_fit();
  }
}

void Index::load(const std::string &prefix, const LoadOptions &opt) {
  *this = Index();
  read_files_(prefix);
  engine_ = (opt.use_amx && amx_available()) ? Engine::kAmx : Engine::kAvx512;
  precision_ = opt.precision;
  if (precision_ == Precision::kBf16) {
    build_bf16_();
    raw_.clear();
    return;
  }
  uint32_t stored = 0;
  const bool has_dimc = read_rotation_(prefix, &stored);
  rotate_lists_();
  quantize_sq16_();
  raw_.clear();
  const uint32_t dims = opt.prefix_dims ? opt.prefix_dims : stored;
  if (has_dimc && dims) build_head_(dims);
  if (opt.batched_coarse) build_coarse_planes_();
  id_list_.assign(npts_, UINT32_MAX);
  id_offset_.assign(npts_, UINT32_MAX);
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t c = 0; c < static_cast<int64_t>(nlist_); ++c) {
    const auto &ids = ids_[c];
    for (uint32_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < npts_) {
        id_list_[ids[i]] = static_cast<uint32_t>(c);
        id_offset_[ids[i]] = i;
      }
    }
  }
}

std::vector<uint32_t> Index::list_sizes() const {
  std::vector<uint32_t> n;
  n.reserve(ids_.size());
  for (const auto &v : ids_) n.push_back(static_cast<uint32_t>(v.size()));
  return n;
}

size_t Index::memory_bytes() const {
  auto nested = [](const auto &v) {
    size_t s = 0;
    for (const auto &x : v) s += x.size() * sizeof(x[0]);
    return s;
  };
  return nested(ids_) + nested(hi_) + nested(lo_) + nested(norm_) + nested(hi_norm_) +
         nested(head_) + nested(head_norm_) + nested(bf16_) + nested(bf16_norm_) +
         centroids_.size() * sizeof(float) + rotation_.size() * sizeof(float) +
         cen_hi_.size() + cen_lo_.size() + cen_norm_.size() * sizeof(int64_t) +
         (id_list_.size() + id_offset_.size()) * sizeof(uint32_t);
}

// ---------------------------------------------------------------------------
// Search

void Index::coarse_flat_(const float *q, uint32_t nprobe, uint32_t *out) const {
  std::priority_queue<PFI> heap;
  for (uint32_t c = 0; c < nlist_; ++c) {
    const float d = l2_fp32(q, centroids_.data() + static_cast<size_t>(c) * dim_, dim_);
    if (heap.size() < nprobe) heap.emplace(d, c);
    else if (d < heap.top().first) { heap.pop(); heap.emplace(d, c); }
  }
  std::vector<PFI> tmp;
  tmp.reserve(heap.size());
  while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
  std::reverse(tmp.begin(), tmp.end());
  for (uint32_t i = 0; i < tmp.size(); ++i) out[i] = tmp[i].second;
}

void Index::coarse_batched_sq16_(const uint8_t *qhi, const uint8_t *qlo,
                                 const int64_t *qnorm, uint32_t B,
                                 uint32_t nprobe, uint32_t *probed) const {
  const uint32_t Dc = dc_;
  const uint32_t np = std::min<uint32_t>(nprobe, nlist_);
  if (B == 0 || np == 0) return;
#pragma omp parallel
  {
    std::vector<float> drow(static_cast<size_t>(16) * nlist_);
    std::vector<uint32_t> ord(nlist_);
#pragma omp for schedule(dynamic, 1)
    for (int64_t q0 = 0; q0 < static_cast<int64_t>(B); q0 += 16) {
      const uint32_t mm = std::min<uint32_t>(16u, B - static_cast<uint32_t>(q0));
      sq16_l2(engine_, qhi + static_cast<size_t>(q0) * Dc,
              qlo + static_cast<size_t>(q0) * Dc, cen_hi_.data(), cen_lo_.data(),
              qnorm + q0, cen_norm_.data(), sq_step_sq_, mm, nlist_, Dc, drow.data());
      for (uint32_t j = 0; j < mm; ++j) {
        const float *dq = drow.data() + static_cast<size_t>(j) * nlist_;
        std::iota(ord.begin(), ord.end(), 0u);
        if (np < nlist_) {
          std::nth_element(ord.begin(), ord.begin() + np, ord.end(),
                           [&](uint32_t a, uint32_t b) { return dq[a] < dq[b]; });
        }
        std::sort(ord.begin(), ord.begin() + np,
                  [&](uint32_t a, uint32_t b) { return dq[a] < dq[b]; });
        uint32_t *pr = probed + (static_cast<size_t>(q0) + j) * nprobe;
        for (uint32_t p = 0; p < np; ++p) pr[p] = ord[p];
        for (uint32_t p = np; p < nprobe; ++p) pr[p] = ord[np - 1];
      }
    }
  }
}

void Index::search(const float *queries, uint32_t nq, const SearchParams &p,
                   uint32_t *ids, float *dists) const {
  if (nq == 0) return;
  if (nlist_ == 0) throw std::logic_error("search before load");
  if (p.k == 0 || p.nprobe == 0 || p.nprobe > nlist_)
    throw std::invalid_argument("need k >= 1 and 1 <= nprobe <= nlist");
  if (precision_ == Precision::kBf16) {
    if (p.ranking != Ranking::kFull)
      throw std::invalid_argument("bf16 precision supports only Ranking::kFull");
    search_bf16_(queries, nq, p.k, p.nprobe, ids, dists);
    return;
  }
  if (p.ranking == Ranking::kPrefix && head_dims_ == 0)
    throw std::invalid_argument("prefix ranking needs the rotation file (.dimc)");
  const float *fq = queries;
  std::vector<float> qrot;
  if (rotated_) {
    qrot.resize(static_cast<size_t>(nq) * dim_);
    rotate_rows(queries, qrot.data(), nq, dim_, rotation_.data());
    fq = qrot.data();
  }
  if (p.ranking == Ranking::kFull)
    search_sq16_full_(queries, fq, nq, p.k, p.nprobe, ids, dists);
  else
    search_sq16_ranked_(queries, fq, nq, p, p.ranking == Ranking::kPrefix, ids, dists);
}

// Ranking pass over all candidates of the probed lists, by the high bytes
// (of the prefix when `prefix`), keeping S per query; then the exact distance
// of those S, gathered into one packed block per query.
void Index::search_sq16_ranked_(const float *queries, const float *fq, uint32_t B,
                                const SearchParams &prm, bool prefix,
                                uint32_t *out_ids, float *out_dists) const {
  const uint32_t d = dim_, Dc = dc_, Nc = Dc / 64, K = prm.k, nprobe = prm.nprobe;
  const uint32_t NBLK = kBlock;
  const uint32_t S = std::max(prm.survivors, K);
  using P = PFI;

  std::vector<uint8_t> qhi(static_cast<size_t>(B) * Dc, 0);
  std::vector<uint8_t> qlo(static_cast<size_t>(B) * Dc, 0);
  std::vector<int64_t> qnorm(B), qhinorm(B), qh0norm(prefix ? B : 0);
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi) {
    const float *qv = fq + static_cast<size_t>(qi) * d;
    int64_t nrm = 0, nrm_hi = 0;
    for (uint32_t k = 0; k < d; ++k) {
      long r = std::lround((static_cast<double>(qv[k]) - sq_lo_) * sq_inv_step_);
      if (r < 0) r = 0; else if (r > 65535) r = 65535;
      const uint32_t v = static_cast<uint32_t>(r);
      const uint8_t vh = static_cast<uint8_t>(v >> 8);
      qhi[static_cast<size_t>(qi) * Dc + k] = vh;
      qlo[static_cast<size_t>(qi) * Dc + k] = static_cast<uint8_t>(v & 0xFF);
      nrm += static_cast<int64_t>(v) * v;
      nrm_hi += static_cast<int64_t>(vh) * vh;
    }
    qnorm[qi] = nrm;
    qhinorm[qi] = nrm_hi;
    if (prefix) {
      const uint32_t dh = std::min(head_dims_, d);
      const uint8_t *qr = qhi.data() + static_cast<size_t>(qi) * Dc;
      int64_t nh = 0;
      for (uint32_t k = 0; k < dh; ++k) nh += static_cast<int64_t>(qr[k]) * qr[k];
      qh0norm[qi] = nh;
    }
  }

  std::vector<uint32_t> probed(static_cast<size_t>(B) * nprobe);
  if (coarse_batched_) {
    coarse_batched_sq16_(qhi.data(), qlo.data(), qnorm.data(), B, nprobe, probed.data());
  } else {
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi)
      coarse_flat_(queries + static_cast<size_t>(qi) * d, nprobe,
                   probed.data() + static_cast<size_t>(qi) * nprobe);
  }

  const bool sel_local = prm.selection == Selection::kLocal;
  std::vector<float> selk;
  std::vector<uint32_t> seli;
  if (sel_local) {
    const size_t nslot = static_cast<size_t>(B) * nprobe * S;
    selk.assign(nslot, std::numeric_limits<float>::max());
    seli.assign(nslot, UINT32_MAX);
  }

  const uint32_t scanDc = prefix ? head_dims_ : Dc;
  const uint32_t scanNc = scanDc / 64;
  const int Tn = omp_get_max_threads();
  std::vector<std::vector<std::priority_queue<P>>> th_coarse(
      sel_local ? 0 : Tn, std::vector<std::priority_queue<P>>(B));

  // One list per work item. With kLocal, slot range [(qi*np + rank)*S, +S)
  // belongs to one (query, probe rank) pair, hence to one list and one thread.
  struct WorkR {
    uint32_t c;
    std::vector<uint32_t> ql, rk;
  };
  WorkList work;
  std::vector<WorkR> work_ranked;
  if (sel_local) {
    std::unordered_map<uint32_t, std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> c2q;
    c2q.reserve(static_cast<size_t>(B) * nprobe / 4 + 16);
    for (uint32_t qi = 0; qi < B; ++qi) {
      const uint32_t *pr = probed.data() + static_cast<size_t>(qi) * nprobe;
      for (uint32_t p = 0; p < nprobe; ++p) {
        auto &e = c2q[pr[p]];
        e.first.push_back(qi);
        e.second.push_back(p);
      }
    }
    work_ranked.reserve(c2q.size());
    for (auto &kv : c2q)
      work_ranked.push_back(WorkR{kv.first, std::move(kv.second.first),
                                  std::move(kv.second.second)});
  } else {
    work = invert(probed.data(), B, nprobe);
  }
  const int64_t nwork = sel_local ? static_cast<int64_t>(work_ranked.size())
                                  : static_cast<int64_t>(work.size());

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::vector<uint8_t> qbh(static_cast<size_t>(B) * Dc, 0);
    std::vector<int64_t> qhn(B, 0);
    std::vector<float> dbuf(static_cast<size_t>(16) * NBLK, 0.0f);
#pragma omp for schedule(dynamic, 1)
    for (int64_t w = 0; w < nwork; ++w) {
      const uint32_t c = sel_local ? work_ranked[w].c : work[w].first;
      const std::vector<uint32_t> &ql = sel_local ? work_ranked[w].ql : work[w].second;
      const uint32_t Mc = static_cast<uint32_t>(ql.size());
      const uint32_t cs = static_cast<uint32_t>(ids_[c].size());
      if (cs == 0 || Mc == 0) continue;
      for (uint32_t j = 0; j < Mc; ++j) {
        std::memcpy(qbh.data() + static_cast<size_t>(j) * scanDc,
                    qhi.data() + static_cast<size_t>(ql[j]) * Dc, scanDc);
        qhn[j] = prefix ? qh0norm[ql[j]] : qhinorm[ql[j]];
      }
      for (uint32_t nb = 0; nb < cs; nb += NBLK) {
        const uint32_t blk = std::min(NBLK, cs - nb);
        const uint8_t *scan_block = (prefix ? head_[c] : hi_[c]).data() +
                                    static_cast<size_t>(nb / 16) * scanNc * 1024;
        const int64_t *nbn = (prefix ? head_norm_[c] : hi_norm_[c]).data() + nb;
        const uint32_t *blk_ids = ids_[c].data() + nb;
        for (uint32_t m0 = 0; m0 < Mc; m0 += 16) {
          const uint32_t mm = std::min<uint32_t>(16, Mc - m0);
          sq16_high_l2(engine_, qbh.data() + static_cast<size_t>(m0) * scanDc,
                       scan_block, qhn.data() + m0, nbn, sq_step_sq_, mm, blk,
                       scanDc, dbuf.data());
          if (sel_local) {
            const std::vector<uint32_t> &rk = work_ranked[w].rk;
            for (uint32_t j = 0; j < mm; ++j) {
              const size_t slot =
                  (static_cast<size_t>(ql[m0 + j]) * nprobe + rk[m0 + j]) * S;
              select_row_topS_sorted(selk.data() + slot, seli.data() + slot,
                                     dbuf.data() + static_cast<size_t>(j) * blk,
                                     blk_ids, blk, S);
            }
          } else {
            std::vector<std::priority_queue<P>> &myc = th_coarse[tid];
            for (uint32_t j = 0; j < mm; ++j)
              reduce_row_topk(myc[ql[m0 + j]], dbuf.data() + static_cast<size_t>(j) * blk,
                              blk_ids, blk, S);
          }
        }
      }
    }
  }

  const uint32_t StileN = round_up(S, 16u);
#pragma omp parallel
  {
    int64_t qn1[1];
    std::vector<uint8_t> ghi(plane_bytes(StileN, Dc)), glo(plane_bytes(StileN, Dc));
    std::vector<uint32_t> sid(StileN);
    std::vector<int64_t> snorm(StileN);
    std::vector<float> dbufS(StileN);
    std::vector<P> selbuf;
    if (sel_local) selbuf.reserve(static_cast<size_t>(nprobe) * S);
#pragma omp for schedule(dynamic, 16)
    for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi) {
      std::priority_queue<P> cmerged;
      if (sel_local) {
        const size_t base = static_cast<size_t>(qi) * nprobe * S;
        const size_t nslot = static_cast<size_t>(nprobe) * S;
        selbuf.clear();
        for (size_t s2 = 0; s2 < nslot; ++s2)
          if (selk[base + s2] < std::numeric_limits<float>::max())
            selbuf.emplace_back(selk[base + s2], seli[base + s2]);
        if (selbuf.size() > S) {
          std::nth_element(selbuf.begin(), selbuf.begin() + (S - 1), selbuf.end());
          selbuf.resize(S);
        }
        for (const P &e : selbuf) cmerged.push(e);
      } else {
        for (int t = 0; t < Tn; ++t) {
          std::priority_queue<P> &h = th_coarse[t][qi];
          while (!h.empty()) {
            P e = h.top(); h.pop();
            if (cmerged.size() < S) cmerged.push(e);
            else if (e.first < cmerged.top().first) { cmerged.pop(); cmerged.push(e); }
          }
        }
      }
      const uint8_t *q1h = qhi.data() + static_cast<size_t>(qi) * Dc;
      const uint8_t *q1l = qlo.data() + static_cast<size_t>(qi) * Dc;
      qn1[0] = qnorm[qi];
      uint32_t ns = 0;
      const uint32_t ntiles = (S + 15u) / 16u;
      std::memset(ghi.data(), 0, static_cast<size_t>(ntiles) * Nc * 1024);
      std::memset(glo.data(), 0, static_cast<size_t>(ntiles) * Nc * 1024);
      while (!cmerged.empty()) {
        const uint32_t id = cmerged.top().second;
        cmerged.pop();
        const uint32_t c = id_list_[id];
        const uint32_t off = id_offset_[id];
        if (c == UINT32_MAX || off == UINT32_MAX) continue;
        if (off >= ids_[c].size()) continue;
        const uint32_t s_tile = off / 16, s_row = off % 16;
        const uint32_t d_tile = ns / 16, d_row = ns % 16;
        const uint8_t *shp = hi_[c].data() + static_cast<size_t>(s_tile) * Nc * 1024;
        const uint8_t *slp = lo_[c].data() + static_cast<size_t>(s_tile) * Nc * 1024;
        uint8_t *dhp = ghi.data() + static_cast<size_t>(d_tile) * Nc * 1024;
        uint8_t *dlp = glo.data() + static_cast<size_t>(d_tile) * Nc * 1024;
        for (uint32_t nc = 0; nc < Nc; ++nc) {
          for (uint32_t hcv = 0; hcv < 16; ++hcv) {
            const size_t soff = ((static_cast<size_t>(nc) * 16 + hcv) * 16 + s_row) * 4;
            const size_t doff = ((static_cast<size_t>(nc) * 16 + hcv) * 16 + d_row) * 4;
            std::memcpy(dhp + doff, shp + soff, 4);
            std::memcpy(dlp + doff, slp + soff, 4);
          }
        }
        sid[ns] = id;
        snorm[ns] = norm_[c][off];
        ++ns;
      }
      uint32_t *ids = out_ids + static_cast<size_t>(qi) * K;
      float *dists = out_dists + static_cast<size_t>(qi) * K;
      std::priority_queue<P> fin;
      if (ns > 0) {
        sq16_l2(engine_, q1h, q1l, ghi.data(), glo.data(), qn1, snorm.data(),
                sq_step_sq_, 1u, ns, Dc, dbufS.data());
        for (uint32_t s = 0; s < ns; ++s) {
          const float dd = dbufS[s];
          if (fin.size() < K) fin.emplace(dd, sid[s]);
          else if (dd < fin.top().first) { fin.pop(); fin.emplace(dd, sid[s]); }
        }
      }
      emit_topk(fin, K, ids, dists);
    }
  }
}

// Exact distance of every candidate (four products per pair).
void Index::search_sq16_full_(const float *queries, const float *fq, uint32_t B,
                              uint32_t K, uint32_t nprobe, uint32_t *out_ids,
                              float *out_dists) const {
  const uint32_t d = dim_, Dc = dc_, Nc = Dc / 64, NBLK = kBlock;
  using P = PFI;
  std::vector<uint8_t> qhi(static_cast<size_t>(B) * Dc, 0);
  std::vector<uint8_t> qlo(static_cast<size_t>(B) * Dc, 0);
  std::vector<int64_t> qnorm(B);
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi) {
    const float *qv = fq + static_cast<size_t>(qi) * d;
    int64_t nrm = 0;
    for (uint32_t k = 0; k < d; ++k) {
      long r = std::lround((static_cast<double>(qv[k]) - sq_lo_) * sq_inv_step_);
      if (r < 0) r = 0; else if (r > 65535) r = 65535;
      const uint32_t v = static_cast<uint32_t>(r);
      qhi[static_cast<size_t>(qi) * Dc + k] = static_cast<uint8_t>(v >> 8);
      qlo[static_cast<size_t>(qi) * Dc + k] = static_cast<uint8_t>(v & 0xFF);
      nrm += static_cast<int64_t>(v) * v;
    }
    qnorm[qi] = nrm;
  }
  std::vector<uint32_t> probed(static_cast<size_t>(B) * nprobe);
  if (coarse_batched_) {
    coarse_batched_sq16_(qhi.data(), qlo.data(), qnorm.data(), B, nprobe, probed.data());
  } else {
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi)
      coarse_flat_(queries + static_cast<size_t>(qi) * d, nprobe,
                   probed.data() + static_cast<size_t>(qi) * nprobe);
  }
  WorkList work = invert(probed.data(), B, nprobe);

  const int Tn = omp_get_max_threads();
  std::vector<std::vector<std::priority_queue<P>>> th_heaps(
      Tn, std::vector<std::priority_queue<P>>(B));
#pragma omp parallel
  {
    std::vector<std::priority_queue<P>> &myheaps = th_heaps[omp_get_thread_num()];
    std::vector<uint8_t> qbh(static_cast<size_t>(B) * Dc);
    std::vector<uint8_t> qbl(static_cast<size_t>(B) * Dc);
    std::vector<int64_t> qn(B);
    std::vector<float> dbuf(static_cast<size_t>(16) * NBLK);
#pragma omp for schedule(dynamic, 1)
    for (int64_t w = 0; w < static_cast<int64_t>(work.size()); ++w) {
      const uint32_t c = work[w].first;
      const std::vector<uint32_t> &ql = work[w].second;
      const uint32_t Mc = static_cast<uint32_t>(ql.size());
      const uint32_t cs = static_cast<uint32_t>(ids_[c].size());
      if (cs == 0 || Mc == 0) continue;
      for (uint32_t j = 0; j < Mc; ++j) {
        std::memcpy(qbh.data() + static_cast<size_t>(j) * Dc,
                    qhi.data() + static_cast<size_t>(ql[j]) * Dc, Dc);
        std::memcpy(qbl.data() + static_cast<size_t>(j) * Dc,
                    qlo.data() + static_cast<size_t>(ql[j]) * Dc, Dc);
        qn[j] = qnorm[ql[j]];
      }
      for (uint32_t nb = 0; nb < cs; nb += NBLK) {
        const uint32_t blk = std::min(NBLK, cs - nb);
        const size_t pkoff = static_cast<size_t>(nb / 16) * Nc * 1024;
        for (uint32_t m0 = 0; m0 < Mc; m0 += 16) {
          const uint32_t mm = std::min<uint32_t>(16, Mc - m0);
          sq16_l2(engine_, qbh.data() + static_cast<size_t>(m0) * Dc,
                  qbl.data() + static_cast<size_t>(m0) * Dc, hi_[c].data() + pkoff,
                  lo_[c].data() + pkoff, qn.data() + m0, norm_[c].data() + nb,
                  sq_step_sq_, mm, blk, Dc, dbuf.data());
          const uint32_t *blk_ids = ids_[c].data() + nb;
          for (uint32_t j = 0; j < mm; ++j)
            reduce_row_topk(myheaps[ql[m0 + j]], dbuf.data() + static_cast<size_t>(j) * blk,
                            blk_ids, blk, K);
        }
      }
    }
  }
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi) {
    std::priority_queue<P> merged;
    for (int t = 0; t < Tn; ++t) {
      std::priority_queue<P> &h = th_heaps[t][qi];
      while (!h.empty()) {
        P e = h.top(); h.pop();
        if (merged.size() < K) merged.push(e);
        else if (e.first < merged.top().first) { merged.pop(); merged.push(e); }
      }
    }
    emit_topk(merged, K, out_ids + static_cast<size_t>(qi) * K,
              out_dists + static_cast<size_t>(qi) * K);
  }
}

void Index::search_bf16_(const float *queries, uint32_t B, uint32_t K,
                         uint32_t nprobe, uint32_t *out_ids, float *out_dists) const {
  const uint32_t d = dim_, dp = round_up(d, 32u), Ncbf = dp / 32, NBLK = kBlock;
  using P = PFI;
  std::vector<uint32_t> probed(static_cast<size_t>(B) * nprobe);
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi)
    coarse_flat_(queries + static_cast<size_t>(qi) * d, nprobe,
                 probed.data() + static_cast<size_t>(qi) * nprobe);
  std::vector<uint16_t> qbf(static_cast<size_t>(B) * dp, 0);
  std::vector<float> qnorm(B);
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi) {
    uint16_t *qb16 = qbf.data() + static_cast<size_t>(qi) * dp;
    to_bf16(queries + static_cast<size_t>(qi) * d, qb16, d);
    qnorm[qi] = bf16_norm(qb16, d);
  }
  WorkList work = invert(probed.data(), B, nprobe);

  const int Tn = omp_get_max_threads();
  std::vector<std::vector<std::priority_queue<P>>> th_heaps(
      Tn, std::vector<std::priority_queue<P>>(B));
#pragma omp parallel
  {
    std::vector<std::priority_queue<P>> &myheaps = th_heaps[omp_get_thread_num()];
    std::vector<uint16_t> qb(static_cast<size_t>(B) * dp);
    std::vector<float> qn(B);
    std::vector<float> dbuf(static_cast<size_t>(16) * NBLK);
#pragma omp for schedule(dynamic, 1)
    for (int64_t w = 0; w < static_cast<int64_t>(work.size()); ++w) {
      const uint32_t c = work[w].first;
      const std::vector<uint32_t> &ql = work[w].second;
      const uint32_t Mc = static_cast<uint32_t>(ql.size());
      const uint32_t cs = static_cast<uint32_t>(ids_[c].size());
      if (cs == 0 || Mc == 0) continue;
      for (uint32_t j = 0; j < Mc; ++j) {
        std::memcpy(qb.data() + static_cast<size_t>(j) * dp,
                    qbf.data() + static_cast<size_t>(ql[j]) * dp,
                    static_cast<size_t>(dp) * sizeof(uint16_t));
        qn[j] = qnorm[ql[j]];
      }
      for (uint32_t nb = 0; nb < cs; nb += NBLK) {
        const uint32_t blk = std::min(NBLK, cs - nb);
        const uint16_t *x = engine_ == Engine::kAmx
                                ? bf16_[c].data() + static_cast<size_t>(nb / 16) * Ncbf * 512
                                : bf16_[c].data() + static_cast<size_t>(nb) * dp;
        for (uint32_t m0 = 0; m0 < Mc; m0 += 16) {
          const uint32_t mm = std::min<uint32_t>(16, Mc - m0);
          bf16_l2(engine_, qb.data() + static_cast<size_t>(m0) * dp, x, qn.data() + m0,
                  bf16_norm_[c].data() + nb, mm, blk, dp, dbuf.data());
          const uint32_t *blk_ids = ids_[c].data() + nb;
          for (uint32_t j = 0; j < mm; ++j)
            reduce_row_topk(myheaps[ql[m0 + j]], dbuf.data() + static_cast<size_t>(j) * blk,
                            blk_ids, blk, K);
        }
      }
    }
  }
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t qi = 0; qi < static_cast<int64_t>(B); ++qi) {
    std::priority_queue<P> merged;
    for (int t = 0; t < Tn; ++t) {
      std::priority_queue<P> &h = th_heaps[t][qi];
      while (!h.empty()) {
        P e = h.top(); h.pop();
        if (merged.size() < K) merged.push(e);
        else if (e.first < merged.top().first) { merged.pop(); merged.push(e); }
      }
    }
    emit_topk(merged, K, out_ids + static_cast<size_t>(qi) * K,
              out_dists + static_cast<size_t>(qi) * K);
  }
}

}  // namespace katoann
