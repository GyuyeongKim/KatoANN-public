#pragma once

#include <cstddef>
#include <cstdint>

namespace katoann {

// Distance kernels on Intel AMX tiles and on AVX-512 VNNI.
//
// A database vector is a 16-bit scalar-quantized code per dimension, stored
// as a high byte plane and a low byte plane. The product of two codes is
// recovered exactly from four 8-bit products,
//   q.x = 65536 (qh.xh) + 256 (qh.xl + ql.xh) + ql.xl,
// so the squared L2 between codes is exact integer arithmetic. Planes are
// padded to a multiple of 64 dimensions and packed in the tile order of
// pack_plane: per group of 16 vectors and 64 dimensions, 16 rows of
// 4 dimensions by 16 vectors.

enum class Engine { kAmx, kAvx512 };

// True if the CPU has AMX and Linux grants this process the tile state.
bool amx_available();

constexpr uint32_t kMaxDim = 4096;

inline size_t plane_bytes(uint32_t n, uint32_t padded_dim) {
  return static_cast<size_t>((n + 15u) / 16u) * (padded_dim / 64u) * 1024;
}

// rows: n x padded_dim bytes, row-major.
void pack_plane(const uint8_t *rows, uint32_t n, uint32_t padded_dim,
                uint8_t *out);

// Squared L2 of m <= 16 queries against n packed database vectors, from the
// full codes. qnorm and xnorm are the integer squared norms of the codes.
// out is m x n, scaled by step_sq.
void sq16_l2(Engine engine, const uint8_t *qhi, const uint8_t *qlo,
             const uint8_t *xhi, const uint8_t *xlo, const int64_t *qnorm,
             const int64_t *xnorm, double step_sq, uint32_t m, uint32_t n,
             uint32_t padded_dim, float *out);

// Same, from the high bytes only (one product instead of four). qnorm and
// xnorm are the squared norms of the high bytes over the scanned dimensions.
void sq16_high_l2(Engine engine, const uint8_t *qhi, const uint8_t *xhi,
                  const int64_t *qnorm, const int64_t *xnorm, double step_sq,
                  uint32_t m, uint32_t n, uint32_t padded_dim, float *out);

// BF16 vectors, for comparison with the 16-bit codes.
inline size_t bf16_packed_elems(uint32_t n, uint32_t dim) {
  return static_cast<size_t>((n + 15u) / 16u) * ((dim + 31u) / 32u) * 512;
}
void to_bf16(const float *src, uint16_t *dst, size_t n);
float bf16_norm(const uint16_t *v, uint32_t dim);
void pack_bf16(const uint16_t *rows, uint32_t n, uint32_t dim, uint16_t *out);

// AMX reads packed rows (pack_bf16); AVX-512 reads row-major rows.
void bf16_l2(Engine engine, const uint16_t *q, const uint16_t *x,
             const float *qnorm, const float *xnorm, uint32_t m, uint32_t n,
             uint32_t dim, float *out);

}  // namespace katoann
