#include "kmeans.h"

#include <cblas.h>
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>

namespace katoann::kmeans {

float l2sq(const float *pX, const float *pY, uint32_t length) {
  const float *pEnd16 = pX + ((length >> 4) << 4);
  const float *pEnd8 = pX + ((length >> 3) << 3);
  const float *pEnd4 = pX + ((length >> 2) << 2);
  const float *pEnd1 = pX + length;
#if defined(__AVX512F__)
  __m512 diff512 = _mm512_setzero_ps();
  while (pX < pEnd16) {
    __m512 d = _mm512_sub_ps(_mm512_loadu_ps(pX), _mm512_loadu_ps(pY));
    diff512 = _mm512_add_ps(diff512, _mm512_mul_ps(d, d));
    pX += 16;
    pY += 16;
  }
  __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512),
                                 _mm512_extractf32x8_ps(diff512, 1));
  while (pX < pEnd8) {
    __m256 d = _mm256_sub_ps(_mm256_loadu_ps(pX), _mm256_loadu_ps(pY));
    diff256 = _mm256_add_ps(diff256, _mm256_mul_ps(d, d));
    pX += 8;
    pY += 8;
  }
  __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256),
                              _mm256_extractf128_ps(diff256, 1));
  while (pX < pEnd4) {
    __m128 d = _mm_sub_ps(_mm_loadu_ps(pX), _mm_loadu_ps(pY));
    diff128 = _mm_add_ps(diff128, _mm_mul_ps(d, d));
    pX += 4;
    pY += 4;
  }
  alignas(16) float t[4];
  _mm_store_ps(t, diff128);
  float diff = t[0] + t[1] + t[2] + t[3];
#else
  (void) pEnd16; (void) pEnd8; (void) pEnd4;
  float diff = 0.0f;
#endif
  while (pX < pEnd1) {
    float c1 = (*pX++) - (*pY++);
    diff += c1 * c1;
  }
  return diff;
}

namespace {

struct Pivot {
  size_t id;
  float dist;
  bool operator<(const Pivot &p) const { return p.dist < dist; }
};

void squared_norms(float *out, const float *data, size_t n, size_t dim) {
#pragma omp parallel for schedule(static, 8192)
  for (int64_t i = 0; i < static_cast<int64_t>(n); i++)
    out[i] = cblas_sdot(dim, data + i * dim, 1, data + i * dim, 1);
}

float elkan_iter(const float *data, size_t num_points, size_t dim,
                 float *centers, size_t num_centers,
                 std::vector<size_t> *closest_docs, uint32_t *closest_center,
                 float *lower_bound, float *upper_bound, float *s_dist,
                 float *half_center_dist, float *new_center_dist,
                 float *new_centers, bool *r) {
  for (size_t c = 0; c < num_centers; ++c) closest_docs[c].clear();

#pragma omp parallel for schedule(static, 1)
  for (int64_t j = 0; j < static_cast<int64_t>(num_centers); j++) {
    s_dist[j] = std::numeric_limits<float>::max();
    for (size_t k = j + 1; k < num_centers; k++) {
      float distance = sqrtf(l2sq(centers + j * dim, centers + k * dim, dim));
      half_center_dist[j * num_centers + k] = 0.5f * distance;
      half_center_dist[k * num_centers + j] = 0.5f * distance;
    }
  }

#pragma omp parallel for schedule(static, 1)
  for (size_t j = 0; j < num_centers; j++) {
    float min_half_dist = std::numeric_limits<float>::max();
    for (size_t k = 0; k < num_centers; k++) {
      if (j == k) continue;
      if (half_center_dist[j * num_centers + k] < min_half_dist)
        min_half_dist = half_center_dist[j * num_centers + k];
    }
    s_dist[j] = min_half_dist;
  }

#pragma omp parallel for schedule(static, 4096)
  for (int64_t j = 0; j < static_cast<int64_t>(num_points); j++) {
    uint32_t current = closest_center[j];
    if (upper_bound[j] <= s_dist[current]) continue;
    float upper_sq;
    if (r[j]) {
      upper_sq = l2sq(data + j * dim, centers + current * dim, dim);
      upper_bound[j] = sqrtf(upper_sq);
      r[j] = false;
      if (upper_bound[j] <= s_dist[current]) continue;
    } else {
      upper_sq = upper_bound[j] * upper_bound[j];
    }
    for (size_t k = 0; k < num_centers; k++) {
      if (k == current) continue;
      if (upper_bound[j] <= lower_bound[j * num_centers + k]) continue;
      if (upper_bound[j] <= half_center_dist[current * num_centers + k]) continue;
      float dist_k_sq = l2sq(data + j * dim, centers + k * dim, dim);
      lower_bound[j * num_centers + k] = sqrtf(dist_k_sq);
      if (dist_k_sq < upper_sq) {
        current = static_cast<uint32_t>(k);
        upper_sq = dist_k_sq;
        upper_bound[j] = lower_bound[j * num_centers + k];
      }
    }
    closest_center[j] = current;
  }

  for (size_t j = 0; j < num_points; j++) closest_docs[closest_center[j]].push_back(j);

  std::vector<double> sums(num_centers * dim, 0.0);
#pragma omp parallel for schedule(static, 1)
  for (int64_t c = 0; c < static_cast<int64_t>(num_centers); ++c) {
    if (closest_docs[c].empty()) continue;
    double *sum = sums.data() + c * dim;
    for (const auto &doc : closest_docs[c]) {
      const float *v = data + doc * dim;
      for (size_t d = 0; d < dim; d++) sum[d] += static_cast<double>(v[d]);
    }
  }

#pragma omp parallel for schedule(static, 1)
  for (int64_t c = 0; c < static_cast<int64_t>(num_centers); ++c) {
    if (!closest_docs[c].empty()) {
      float *center = new_centers + c * dim;
      const double *sum = sums.data() + c * dim;
      const double size = static_cast<double>(closest_docs[c].size());
      for (size_t i = 0; i < dim; i++) center[i] = static_cast<float>(sum[i] / size);
    } else {
      std::memcpy(new_centers + c * dim, centers + c * dim, dim * sizeof(float));
    }
  }

#pragma omp parallel for schedule(static, 1)
  for (int64_t j = 0; j < static_cast<int64_t>(num_centers); j++)
    new_center_dist[j] = sqrtf(l2sq(centers + j * dim, new_centers + j * dim, dim));

#pragma omp parallel for schedule(static, 8192)
  for (int64_t j = 0; j < static_cast<int64_t>(num_points); j++) {
    for (size_t k = 0; k < num_centers; k++) {
      float lb = lower_bound[j * num_centers + k] - new_center_dist[k];
      lower_bound[j * num_centers + k] = lb > 0.0f ? lb : 0.0f;
    }
  }

#pragma omp parallel for schedule(static, 8192)
  for (int64_t j = 0; j < static_cast<int64_t>(num_points); j++) {
    upper_bound[j] += new_center_dist[closest_center[j]];
    r[j] = true;
  }

  std::memcpy(centers, new_centers, sizeof(float) * num_centers * dim);

  const size_t pad = 32, chunk_size = 2 * 8192;
  const size_t nchunks = (num_points + chunk_size - 1) / chunk_size;
  std::vector<float> residuals(nchunks * pad, 0.0f);
#pragma omp parallel for schedule(static, 32)
  for (int64_t chunk = 0; chunk < static_cast<int64_t>(nchunks); ++chunk) {
    for (size_t d = chunk * chunk_size; d < num_points && d < (chunk + 1) * chunk_size; ++d)
      residuals[chunk * pad] +=
          l2sq(data + d * dim, centers + static_cast<size_t>(closest_center[d]) * dim, dim);
  }
  float residual = 0.0f;
  for (size_t chunk = 0; chunk < nchunks; ++chunk) residual += residuals[chunk * pad];
  return residual;
}

}  // namespace

void closest_centers(const float *data, size_t num_points, size_t dim,
                     const float *centers, size_t num_centers, size_t k,
                     uint32_t *out) {
  std::vector<float> point_norms(num_points), center_norms(num_centers);
  squared_norms(point_norms.data(), data, num_points, dim);
  squared_norms(center_norms.data(), centers, num_centers, dim);
  const size_t block = std::min<size_t>(8192, num_points);
  const size_t nblocks = (num_points + block - 1) / block;
  std::vector<float> ones(std::max(num_centers, block), 1.0f);
#pragma omp parallel
  {
    std::vector<float> dist(num_centers * block);
#pragma omp for schedule(dynamic, 1)
    for (size_t b = 0; b < nblocks; b++) {
      const size_t start = b * block;
      const size_t nb = std::min(block, num_points - start);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nb, num_centers, 1, 1.0f,
                  point_norms.data() + start, 1, ones.data(), 1, 0.0f, dist.data(),
                  num_centers);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nb, num_centers, 1, 1.0f,
                  ones.data(), 1, center_norms.data(), 1, 1.0f, dist.data(), num_centers);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nb, num_centers, dim, -2.0f,
                  data + start * dim, dim, centers, dim, 1.0f, dist.data(), num_centers);
      for (size_t i = 0; i < nb; i++) {
        const float *d = dist.data() + i * num_centers;
        if (k == 1) {
          float best = std::numeric_limits<float>::max();
          uint32_t best_id = 0;
          for (size_t j = 0; j < num_centers; j++) {
            if (d[j] < best) {
              best_id = static_cast<uint32_t>(j);
              best = d[j];
            }
          }
          out[start + i] = best_id;
        } else {
          std::priority_queue<Pivot> q;
          for (size_t j = 0; j < num_centers; j++) q.push(Pivot{j, d[j]});
          for (size_t j = 0; j < k; j++) {
            out[(start + i) * k + j] = static_cast<uint32_t>(q.top().id);
            q.pop();
          }
        }
      }
    }
  }
}

float elkan(const float *data, size_t num_points, size_t dim, float *centers,
            size_t num_centers, size_t max_iters) {
  std::vector<std::vector<size_t>> closest_docs(num_centers);
  std::vector<uint32_t> closest_center(num_points);
  std::vector<float> lower_bound(num_points * num_centers, 0.0f);
  std::vector<float> upper_bound(num_points), s_dist(num_centers);
  std::vector<float> half_center_dist(num_centers * num_centers);
  std::vector<float> new_center_dist(num_centers), new_centers(num_centers * dim);
  std::unique_ptr<bool[]> r(new bool[num_points]());

  for (size_t j = 0; j < num_points; j++) {
    float best = std::numeric_limits<float>::max();
    uint32_t best_id = 0;
    for (size_t k = 0; k < num_centers; k++) {
      float d = l2sq(data + j * dim, centers + k * dim, dim);
      if (d < best) {
        best = d;
        best_id = static_cast<uint32_t>(k);
      }
    }
    upper_bound[j] = sqrtf(best);
    closest_center[j] = best_id;
  }

  float residual = std::numeric_limits<float>::max();
  for (size_t i = 0; i < max_iters; ++i) {
    const float old_residual = residual;
    residual = elkan_iter(data, num_points, dim, centers, num_centers,
                          closest_docs.data(), closest_center.data(),
                          lower_bound.data(), upper_bound.data(), s_dist.data(),
                          half_center_dist.data(), new_center_dist.data(),
                          new_centers.data(), r.get());
    if (i > 0 && residual > std::numeric_limits<float>::epsilon()) {
      if (std::fabs((old_residual - residual) / residual) < 1e-5) break;
    } else if (residual < std::numeric_limits<float>::epsilon()) {
      break;
    }
  }
  return residual;
}

void kmeanspp(const float *data, size_t num_points, size_t dim, float *centers,
              size_t num_centers, uint64_t seed) {
  std::mt19937 gen(seed ? static_cast<std::mt19937::result_type>(seed)
                        : std::random_device{}());
  std::uniform_real_distribution<> uniform(0, 1);
  std::uniform_int_distribution<size_t> pick(0, num_points - 1);
  std::vector<size_t> picked;
  size_t first = pick(gen);
  picked.push_back(first);
  std::memcpy(centers, data + first * dim, dim * sizeof(float));

  std::vector<float> dist(num_points);
#pragma omp parallel for schedule(static, 4096)
  for (int64_t i = 0; i < static_cast<int64_t>(num_points); i++)
    dist[i] = l2sq(data + i * dim, data + first * dim, dim);

  size_t num_picked = 1;
  bool all_zero = false;
  while (num_picked < num_centers) {
    double dart = uniform(gen);
    double sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(static, 8192)
    for (int64_t i = 0; i < static_cast<int64_t>(num_points); i++) sum += dist[i];
    if (sum == 0) all_zero = true;
    dart *= sum;
    size_t next = 0;
    double prefix = 0;
    for (size_t i = 0; i < num_points; i++) {
      next = i;
      if (dart >= prefix && dart < prefix + static_cast<double>(dist[i])) break;
      prefix += static_cast<double>(dist[i]);
    }
    if (std::find(picked.begin(), picked.end(), next) != picked.end() && !all_zero)
      continue;
    picked.push_back(next);
    std::memcpy(centers + num_picked * dim, data + next * dim, dim * sizeof(float));
#pragma omp parallel for schedule(static, 4096)
    for (int64_t i = 0; i < static_cast<int64_t>(num_points); i++)
      dist[i] = std::min(dist[i], l2sq(data + i * dim, data + next * dim, dim));
    num_picked++;
  }
}

}  // namespace katoann::kmeans
