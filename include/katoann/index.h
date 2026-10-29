#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "katoann/kernels.h"

namespace katoann {

// An inverted-file (IVF) index over float vectors with squared L2 distance.
// For cosine similarity, normalize the base and query vectors first.
//
// Files of an index with prefix P:
//   P.meta       dimension, list count, vector count, metric, normalized flag
//   P.centroids  list centroids, float32
//   P.invlist    per list: size, vector ids, float32 vectors
//   P.norms      per list: size (kept for compatibility)
//   P.dimc       orthogonal PCA rotation and a packed prefix of the rotated
//                high bytes (the prefix is rebuilt at load)

struct BuildOptions {
  uint32_t nlist = 2048;
  uint32_t train_size = 256000;
  uint32_t kmeans_iters = 12;
  uint64_t seed = 0;              // k-means++ seed; 0 picks a random seed
  bool normalized = false;        // record that the vectors are unit-length
  uint32_t rotation_sample = 100000;
  uint32_t jacobi_sweeps = 8;
};

enum class Precision {
  kSq16,  // 16-bit codes in high and low byte planes (default)
  kBf16,  // bfloat16 vectors, full scan, per-query coarse search
};

struct LoadOptions {
  Precision precision = Precision::kSq16;
  bool use_amx = true;            // AVX-512 VNNI when false or unavailable
  uint32_t prefix_dims = 0;       // dimensions of the prefix ranking pass;
                                  // 0 takes the value stored in P.dimc
  bool batched_coarse = true;     // coarse search as one matrix product per batch
};

enum class Ranking {
  kFull,        // exact distance of every candidate
  kHighBytes,   // rank by the high bytes, refine the survivors exactly
  kPrefix,      // rank by the high bytes of the leading rotated dimensions
};

enum class Selection {
  kThreshold,   // one running top-S per query, filtered 32 distances at a time
  kLocal,       // a top-S per (query, list), merged after the scan
};

struct SearchParams {
  uint32_t k = 10;
  uint32_t nprobe = 64;
  Ranking ranking = Ranking::kPrefix;
  uint32_t survivors = 32;        // candidates refined per query (at least k)
  Selection selection = Selection::kThreshold;
};

class Index {
 public:
  Index() = default;
  Index(const Index &) = delete;
  Index &operator=(const Index &) = delete;
  Index(Index &&) = default;
  Index &operator=(Index &&) = default;

  // Partitions data (n x dim, row-major) with k-means, writes the index files,
  // and computes the rotation for the prefix ranking pass.
  static void build(const float *data, uint32_t n, uint32_t dim,
                    const BuildOptions &opt, const std::string &prefix);

  void load(const std::string &prefix, const LoadOptions &opt = {});

  // Searches nq queries together. ids and dists are nq x k, nearest first.
  void search(const float *queries, uint32_t nq, const SearchParams &p,
              uint32_t *ids, float *dists) const;

  uint32_t dim() const { return dim_; }
  uint32_t nlist() const { return nlist_; }
  uint32_t size() const { return npts_; }
  bool normalized() const { return normalized_; }
  Engine engine() const { return engine_; }
  Precision precision() const { return precision_; }
  uint32_t prefix_dims() const { return head_dims_; }
  bool rotated() const { return rotated_; }
  bool batched_coarse() const { return coarse_batched_; }
  std::vector<uint32_t> list_sizes() const;
  size_t memory_bytes() const;

 private:
  void read_files_(const std::string &prefix);
  bool read_rotation_(const std::string &prefix, uint32_t *stored_dims);
  void rotate_lists_();
  void quantize_sq16_();
  void build_head_(uint32_t dims);
  void build_coarse_planes_();
  void build_bf16_();
  void write_dimc_(const std::string &prefix) const;
  void compute_rotation_(uint32_t sample, uint32_t sweeps);

  void coarse_flat_(const float *q, uint32_t nprobe, uint32_t *out) const;
  void coarse_batched_sq16_(const uint8_t *qhi, const uint8_t *qlo,
                            const int64_t *qnorm, uint32_t nq, uint32_t nprobe,
                            uint32_t *probed) const;
  void search_sq16_full_(const float *queries, const float *fq, uint32_t nq,
                         uint32_t k, uint32_t nprobe, uint32_t *ids,
                         float *dists) const;
  void search_sq16_ranked_(const float *queries, const float *fq, uint32_t nq,
                           const SearchParams &p, bool prefix, uint32_t *ids,
                           float *dists) const;
  void search_bf16_(const float *queries, uint32_t nq, uint32_t k,
                    uint32_t nprobe, uint32_t *ids, float *dists) const;

  uint32_t dim_ = 0, nlist_ = 0, npts_ = 0;
  bool normalized_ = false;
  Engine engine_ = Engine::kAvx512;
  Precision precision_ = Precision::kSq16;

  std::vector<float> centroids_;
  std::vector<std::vector<uint32_t>> ids_;
  std::vector<std::vector<float>> raw_;  // only while building and loading

  // 16-bit codes: v = round((x - lo) / step), clamped to [0, 65535].
  float sq_lo_ = 0.0f;
  double sq_inv_step_ = 0.0, sq_step_ = 0.0, sq_step_sq_ = 0.0;
  uint32_t dc_ = 0;  // dim padded to a multiple of 64
  std::vector<std::vector<uint8_t>> hi_, lo_;
  std::vector<std::vector<int64_t>> norm_, hi_norm_;

  // Prefix ranking pass on the rotated high bytes.
  bool rotated_ = false;
  std::vector<float> rotation_;  // dim x dim, row j = basis vector j
  uint32_t head_dims_ = 0;
  std::vector<std::vector<uint8_t>> head_;
  std::vector<std::vector<int64_t>> head_norm_;

  bool coarse_batched_ = false;
  std::vector<uint8_t> cen_hi_, cen_lo_;
  std::vector<int64_t> cen_norm_;

  std::vector<uint32_t> id_list_, id_offset_;

  std::vector<std::vector<uint16_t>> bf16_;
  std::vector<std::vector<float>> bf16_norm_;
};

}  // namespace katoann
