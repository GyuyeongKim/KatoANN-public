#include "katoann/kernels.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstring>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace katoann {

namespace {

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

// CPUID.(EAX=7,ECX=0):EDX bit 22 AMX-BF16, bit 24 AMX-TILE, bit 25 AMX-INT8.
bool cpu_has_amx() {
#if defined(__GNUC__) && defined(__x86_64__)
  uint32_t eax = 7, ebx, ecx = 0, edx;
  __asm__ volatile("cpuid"
                   : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                   : "0"(eax), "2"(ecx));
  return ((edx >> 24) & 1) && ((edx >> 25) & 1) && ((edx >> 22) & 1);
#else
  return false;
#endif
}

bool request_tile_permission() {
#if defined(__linux__) && defined(__x86_64__)
  return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
#else
  return false;
#endif
}

std::atomic<int> g_amx_state{-1};

struct alignas(64) TileCfg {
  uint8_t palette;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

// The tile configuration of the calling thread. initial-exec keeps the access a
// plain load in position-independent builds too.
__attribute__((tls_model("initial-exec"))) thread_local uint8_t g_cfg_kind = 0;
__attribute__((tls_model("initial-exec"))) thread_local uint32_t g_cfg_m = 0;

// Tiles 0,1: query high and low bytes (m rows). Tiles 2,3: database high and
// low bytes (16 rows). Tiles 4-7: the four products, m x 16 int32.
void configure_tiles_sq16(uint32_t m) {
#if defined(__AMX_INT8__)
  if (g_cfg_kind == 1 && g_cfg_m == m) return;
  TileCfg cfg{};
  cfg.palette = 1;
  for (int t = 0; t < 8; ++t) cfg.colsb[t] = 64;
  cfg.rows[0] = cfg.rows[1] = static_cast<uint8_t>(m);
  cfg.rows[2] = cfg.rows[3] = 16;
  cfg.rows[4] = cfg.rows[5] = cfg.rows[6] = cfg.rows[7] = static_cast<uint8_t>(m);
  _tile_loadconfig(&cfg);
  g_cfg_kind = 1;
  g_cfg_m = m;
#else
  (void) m;
#endif
}

void configure_tiles_bf16(uint32_t m) {
#if defined(__AMX_BF16__)
  if (g_cfg_kind == 2 && g_cfg_m == m) return;
  TileCfg cfg{};
  cfg.palette = 1;
  cfg.rows[0] = static_cast<uint8_t>(m); cfg.colsb[0] = 64;
  cfg.rows[1] = 16;                      cfg.colsb[1] = 64;
  cfg.rows[2] = static_cast<uint8_t>(m); cfg.colsb[2] = 64;
  _tile_loadconfig(&cfg);
  g_cfg_kind = 2;
  g_cfg_m = m;
#else
  (void) m;
#endif
}

// AVX-512 VNNI with the 16 lanes on 16 database vectors. The database bytes
// are flipped to signed (x - 128) for VPDPBUSD and each sum is corrected by
// 128 times the sum of the query bytes, so the products are exact.
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
template <bool Full>
__attribute__((noinline)) void sq16_avx512(const uint8_t *qhi, const uint8_t *qlo, const uint8_t *nhi_pk,
                 const uint8_t *nlo_pk, const int64_t *query_norms,
                 const int64_t *neigh_norms, double step_sq, uint32_t M,
                 uint32_t N, uint32_t Dc, float *dists) {
  int64_t sumh[16] = {}, suml[16] = {};
  const __m512i zero = _mm512_setzero_si512();
  const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
  for (uint32_t m = 0; m < M; ++m) {
    __m512i sh = zero, sl = zero;
    for (uint32_t k = 0; k < Dc; k += 64) {
      sh = _mm512_add_epi64(sh, _mm512_sad_epu8(
          _mm512_loadu_si512(qhi + static_cast<size_t>(m) * Dc + k), zero));
      if constexpr (Full) sl = _mm512_add_epi64(sl, _mm512_sad_epu8(
          _mm512_loadu_si512(qlo + static_cast<size_t>(m) * Dc + k), zero));
    }
    sumh[m] = _mm512_reduce_add_epi64(sh);
    if constexpr (Full) suml[m] = _mm512_reduce_add_epi64(sl);
  }
  for (uint32_t n0 = 0; n0 < N; n0 += 16) {
    const size_t base = static_cast<size_t>(n0) * Dc;
    for (uint32_t m = 0; m < M; ++m) {
      __m512i ahh[4], ahl[4], alh[4], all_[4];
      for (int g = 0; g < 4; ++g) {
        ahh[g] = zero;
        if constexpr (Full) ahl[g] = alh[g] = all_[g] = zero;
      }
      for (uint32_t k4 = 0; k4 < Dc / 4; ++k4) {
        uint32_t ih = 0, il = 0;
        std::memcpy(&ih, qhi + static_cast<size_t>(m) * Dc + k4 * 4, 4);
        const __m512i qh = _mm512_set1_epi32(ih);
        const __m512i nh = _mm512_xor_si512(
            _mm512_loadu_si512(nhi_pk + base + static_cast<size_t>(k4) * 64), flip);
        const uint32_t g = k4 & 3u;
        ahh[g] = _mm512_dpbusd_epi32(ahh[g], qh, nh);
        if constexpr (Full) {
          std::memcpy(&il, qlo + static_cast<size_t>(m) * Dc + k4 * 4, 4);
          const __m512i ql = _mm512_set1_epi32(il);
          const __m512i nl = _mm512_xor_si512(
              _mm512_loadu_si512(nlo_pk + base + static_cast<size_t>(k4) * 64), flip);
          ahl[g] = _mm512_dpbusd_epi32(ahl[g], qh, nl);
          alh[g] = _mm512_dpbusd_epi32(alh[g], ql, nh);
          all_[g] = _mm512_dpbusd_epi32(all_[g], ql, nl);
        }
      }
      auto sum4 = [](__m512i *v) { return _mm512_add_epi32(
          _mm512_add_epi32(v[0], v[1]), _mm512_add_epi32(v[2], v[3])); };
      alignas(64) int32_t thh[16], thl[16], tlh[16], tll[16];
      _mm512_store_si512(thh, sum4(ahh));
      if constexpr (Full) {
        _mm512_store_si512(thl, sum4(ahl));
        _mm512_store_si512(tlh, sum4(alh));
        _mm512_store_si512(tll, sum4(all_));
      }
      for (uint32_t lane = 0; lane < std::min<uint32_t>(16, N - n0); ++lane) {
        const int64_t phh = static_cast<int64_t>(thh[lane]) + 128 * sumh[m];
        int64_t dot = phh;
        if constexpr (Full) {
          const int64_t phl = static_cast<int64_t>(thl[lane]) + 128 * sumh[m];
          const int64_t plh = static_cast<int64_t>(tlh[lane]) + 128 * suml[m];
          const int64_t pll = static_cast<int64_t>(tll[lane]) + 128 * suml[m];
          dot = (phh << 16) + ((phl + plh) << 8) + pll;
        }
        const int64_t l2 = query_norms[m] + neigh_norms[n0 + lane] - 2 * dot;
        dists[static_cast<size_t>(m) * N + n0 + lane] =
            static_cast<float>(static_cast<double>(l2) * step_sq);
      }
    }
  }
}
#endif

#if defined(__AMX_INT8__)
// The AMX and AVX-512 paths are separate functions, so that neither inlines
// into the other and changes how its epilogue is compiled.
__attribute__((noinline)) void sq16_l2_amx(
    const uint8_t *qhi, const uint8_t *qlo, const uint8_t *nhi_pk, const uint8_t *nlo_pk,
    const int64_t *query_norms, const int64_t *neigh_norms, double step_sq, uint32_t M,
    uint32_t N, uint32_t Dc, float *dists) {
  // The bounds also tell the compiler that M <= 16, which it needs to
  // vectorize the epilogue along the contiguous i dimension.
  if (M == 0 || M > 16 || (Dc % 64) != 0 || N == 0) return;
  configure_tiles_sq16(M);
  const uint32_t Nc = Dc / 64, Nk = (N + 15u) / 16u;
  alignas(64) int32_t chh[256], chl[256], clh[256], cll[256];
  for (uint32_t nk = 0; nk < Nk; ++nk) {
    _tile_zero(4); _tile_zero(5); _tile_zero(6); _tile_zero(7);
    for (uint32_t nc = 0; nc < Nc; ++nc) {
      _tile_loadd(0, qhi + nc * 64, Dc);
      _tile_loadd(1, qlo + nc * 64, Dc);
      _tile_loadd(2, nhi_pk + (static_cast<size_t>(nk) * Nc + nc) * 1024, 64);
      _tile_loadd(3, nlo_pk + (static_cast<size_t>(nk) * Nc + nc) * 1024, 64);
      _tile_dpbuud(4, 0, 2);
      _tile_dpbuud(5, 0, 3);
      _tile_dpbuud(6, 1, 2);
      _tile_dpbuud(7, 1, 3);
    }
    _tile_stored(4, chh, 64); _tile_stored(5, chl, 64);
    _tile_stored(6, clh, 64); _tile_stored(7, cll, 64);
    const uint32_t na = std::min<uint32_t>(16u, N - nk * 16);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t i = 0; i < na; ++i) {
        const int64_t phh = chh[m * 16 + i], phl = chl[m * 16 + i];
        const int64_t plh = clh[m * 16 + i], pll = cll[m * 16 + i];
        const int64_t dot = (phh << 16) + ((phl + plh) << 8) + pll;
        const int64_t l2 = query_norms[m] + neigh_norms[nk * 16 + i] - 2 * dot;
        dists[static_cast<size_t>(m) * N + nk * 16 + i] =
            static_cast<float>(static_cast<double>(l2) * step_sq);
      }
    }
  }
}

__attribute__((noinline)) void sq16_high_l2_amx(
    const uint8_t *qhi, const uint8_t *nhi_pk, const int64_t *query_hi_norms,
    const int64_t *neigh_hi_norms, double step_sq, uint32_t M, uint32_t N, uint32_t Dc,
    float *dists) {
  if (M == 0 || M > 16 || (Dc % 64) != 0 || N == 0) return;
  configure_tiles_sq16(M);
  const uint32_t Nc = Dc / 64, Nk = (N + 15u) / 16u;
  alignas(64) int32_t chh[256];
  for (uint32_t nk = 0; nk < Nk; ++nk) {
    _tile_zero(4);
    for (uint32_t nc = 0; nc < Nc; ++nc) {
      _tile_loadd(0, qhi + nc * 64, Dc);
      _tile_loadd(2, nhi_pk + (static_cast<size_t>(nk) * Nc + nc) * 1024, 64);
      _tile_dpbuud(4, 0, 2);
    }
    _tile_stored(4, chh, 64);
    const uint32_t na = std::min<uint32_t>(16u, N - nk * 16);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t i = 0; i < na; ++i) {
        const int64_t phh = chh[m * 16 + i];
        const int64_t l2 =
            query_hi_norms[m] + neigh_hi_norms[nk * 16 + i] - 2 * phh;
        dists[static_cast<size_t>(m) * N + nk * 16 + i] =
            static_cast<float>(static_cast<double>(l2) * step_sq);
      }
    }
  }
}
#endif

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

bool amx_available() {
  int s = g_amx_state.load(std::memory_order_acquire);
  if (s != -1) return s == 1;
  bool ok = cpu_has_amx() && request_tile_permission();
  g_amx_state.store(ok ? 1 : 0, std::memory_order_release);
  return ok;
}

void pack_plane(const uint8_t *plane, uint32_t N, uint32_t Dc, uint8_t *out) {
  const uint32_t Nc = Dc / 64u;
  std::memset(out, 0, plane_bytes(N, Dc));
  for (uint32_t n = 0; n < N; ++n) {
    const uint32_t nk = n / 16, hk = n % 16;
    const uint8_t *row = plane + static_cast<size_t>(n) * Dc;
    for (uint32_t k = 0; k < Dc; ++k) {
      const uint32_t nc = k / 64, hcv = (k % 64) / 4, v = k % 4;
      out[((((static_cast<size_t>(nk) * Nc + nc) * 16) + hcv) * 16 + hk) * 4 + v] =
          row[k];
    }
  }
}

void sq16_l2(Engine engine, const uint8_t *qhi, const uint8_t *qlo,
             const uint8_t *nhi_pk, const uint8_t *nlo_pk,
             const int64_t *query_norms, const int64_t *neigh_norms,
             double step_sq, uint32_t M, uint32_t N, uint32_t Dc,
             float *dists) {
  if (M == 0 || N == 0) return;
#if defined(__AMX_INT8__)
  if (engine == Engine::kAmx) {
    sq16_l2_amx(qhi, qlo, nhi_pk, nlo_pk, query_norms, neigh_norms, step_sq, M, N, Dc, dists);
    return;
  }
#endif
  (void) engine;
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
  sq16_avx512<true>(qhi, qlo, nhi_pk, nlo_pk, query_norms, neigh_norms,
                    step_sq, M, N, Dc, dists);
#endif
}

void sq16_high_l2(Engine engine, const uint8_t *qhi, const uint8_t *nhi_pk,
                  const int64_t *query_hi_norms, const int64_t *neigh_hi_norms,
                  double step_sq, uint32_t M, uint32_t N, uint32_t Dc,
                  float *dists) {
  if (M == 0 || N == 0) return;
#if defined(__AMX_INT8__)
  if (engine == Engine::kAmx) {
    sq16_high_l2_amx(qhi, nhi_pk, query_hi_norms, neigh_hi_norms, step_sq, M, N, Dc, dists);
    return;
  }
#endif
  (void) engine;
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
  sq16_avx512<false>(qhi, nullptr, nhi_pk, nullptr, query_hi_norms,
                     neigh_hi_norms, step_sq, M, N, Dc, dists);
#endif
}

void to_bf16(const float *src, uint16_t *dst, size_t n) {
#if defined(__AVX512BF16__)
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256bh b = _mm512_cvtneps_pbh(_mm512_loadu_ps(src + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), (__m256i) b);
  }
  for (; i < n; ++i) {
    uint32_t u; std::memcpy(&u, &src[i], 4);
    dst[i] = static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
#else
  for (size_t i = 0; i < n; ++i) {
    uint32_t u; std::memcpy(&u, &src[i], 4);
    dst[i] = static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
#endif
}

float bf16_norm(const uint16_t *vec, uint32_t d) {
  float s = 0.0f;
  for (uint32_t i = 0; i < d; ++i) { float v = bf16_to_f32(vec[i]); s += v * v; }
  return s;
}

void pack_bf16(const uint16_t *data, uint32_t N, uint32_t d, uint16_t *out) {
  const uint32_t Nc = (d + 31u) / 32u;
  std::memset(out, 0, bf16_packed_elems(N, d) * sizeof(uint16_t));
  for (uint32_t n = 0; n < N; ++n) {
    const uint32_t nk = n / 16, ni = n % 16;
    for (uint32_t k = 0; k < d; ++k) {
      const uint32_t nc = k / 32, k2 = (k % 32) / 2, p = k % 2;
      out[((((static_cast<size_t>(nk) * Nc + nc) * 16) + k2) * 16 + ni) * 2 + p] =
          data[static_cast<size_t>(n) * d + k];
    }
  }
}

void bf16_l2(Engine engine, const uint16_t *queries, const uint16_t *neigh,
             const float *query_norms, const float *neigh_norms, uint32_t M,
             uint32_t N, uint32_t d, float *dists) {
#if defined(__AMX_BF16__)
  if (engine == Engine::kAmx) {
    configure_tiles_bf16(M);
    const uint32_t Nc = d / 32, Nk = (N + 15u) / 16u;
    alignas(64) float cbuf[16 * 16];
    for (uint32_t nk = 0; nk < Nk; ++nk) {
      _tile_zero(2);
      for (uint32_t nc = 0; nc < Nc; ++nc) {
        _tile_loadd(0, queries + static_cast<size_t>(nc) * 32, d * 2);
        _tile_loadd(1, neigh + (static_cast<size_t>(nk) * Nc + nc) * 512, 64);
        _tile_dpbf16ps(2, 0, 1);
      }
      _tile_stored(2, cbuf, 64);
      const uint32_t na = std::min<uint32_t>(16, N - nk * 16);
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t i = 0; i < na; ++i)
          dists[static_cast<size_t>(m) * N + nk * 16 + i] =
              query_norms[m] + neigh_norms[nk * 16 + i] - 2.0f * cbuf[m * 16 + i];
    }
    return;
  }
#endif
  (void) engine;
#if defined(__AVX512BF16__)
  alignas(64) uint16_t qpack[16 * kMaxDim];
  std::memset(qpack, 0, static_cast<size_t>(d) * 16 * sizeof(uint16_t));
  for (uint32_t m = 0; m < M; ++m)
    for (uint32_t k = 0; k < d; ++k)
      qpack[static_cast<size_t>(k / 2) * 32 + m * 2 + (k & 1u)] =
          queries[static_cast<size_t>(m) * d + k];
  alignas(64) float qpn16[16] = {};
  for (uint32_t m = 0; m < M; ++m) qpn16[m] = query_norms[m];
  const __m512 qpn_v = _mm512_load_ps(qpn16);
  const uint32_t k2n = d / 2;
  for (uint32_t k = 0; k < N; ++k) {
    const uint16_t *n = neigh + static_cast<size_t>(k) * d;
    __m512 acc[8];
    for (int g = 0; g < 8; ++g) acc[g] = _mm512_setzero_ps();
    for (uint32_t k2 = 0; k2 < k2n; ++k2) {
      __m512bh qv = (__m512bh) _mm512_loadu_si512(
          reinterpret_cast<const __m512i *>(qpack + static_cast<size_t>(k2) * 32));
      int32_t nb;
      std::memcpy(&nb, n + static_cast<size_t>(k2) * 2, 4);
      __m512bh nv = (__m512bh) _mm512_set1_epi32(nb);
      acc[k2 & 7] = _mm512_dpbf16_ps(acc[k2 & 7], nv, qv);
    }
    for (int s = 4; s > 0; s >>= 1)
      for (int g = 0; g < s; ++g) acc[g] = _mm512_add_ps(acc[g], acc[g + s]);
    __m512 dot = acc[0];
    __m512 l2 = _mm512_sub_ps(
        _mm512_add_ps(qpn_v, _mm512_set1_ps(neigh_norms[k])),
        _mm512_add_ps(dot, dot));
    alignas(64) float t[16];
    _mm512_store_ps(t, l2);
    for (uint32_t m = 0; m < M; ++m) dists[static_cast<size_t>(m) * N + k] = t[m];
  }
#else
  (void) queries; (void) neigh; (void) query_norms; (void) neigh_norms;
  (void) M; (void) N; (void) d; (void) dists;
#endif
}

}  // namespace katoann
