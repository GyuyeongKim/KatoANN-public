// The SQ16 kernels on AMX and on AVX-512 must equal the scalar int64 formula
// exactly, including partial tiles and the extreme byte values.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "katoann/kernels.h"

using namespace katoann;

namespace {

int failures = 0;

void check(bool ok, const char *what, uint32_t m, uint32_t n, uint32_t dc) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s m=%u n=%u dims=%u\n", what, m, n, dc);
  }
}

void run_case(std::mt19937 &rng, uint32_t m, uint32_t n, uint32_t dc, bool amx) {
  std::uniform_int_distribution<int> byte(0, 255), coin(0, 3);
  auto draw = [&]() {
    const int c = coin(rng);
    return static_cast<uint8_t>(c == 0 ? 0 : c == 1 ? 255 : byte(rng));
  };
  std::vector<uint8_t> qh(m * dc), ql(m * dc), xh(static_cast<size_t>(n) * dc),
      xl(static_cast<size_t>(n) * dc);
  for (auto &v : qh) v = draw();
  for (auto &v : ql) v = draw();
  for (auto &v : xh) v = draw();
  for (auto &v : xl) v = draw();
  std::vector<int64_t> qn(m), qhn(m), xn(n), xhn(n);
  auto code = [](uint8_t h, uint8_t l) { return static_cast<int64_t>(h) * 256 + l; };
  for (uint32_t i = 0; i < m; ++i) {
    qn[i] = qhn[i] = 0;
    for (uint32_t k = 0; k < dc; ++k) {
      const int64_t c = code(qh[i * dc + k], ql[i * dc + k]);
      qn[i] += c * c;
      qhn[i] += static_cast<int64_t>(qh[i * dc + k]) * qh[i * dc + k];
    }
  }
  for (uint32_t j = 0; j < n; ++j) {
    xn[j] = xhn[j] = 0;
    for (uint32_t k = 0; k < dc; ++k) {
      const int64_t c = code(xh[static_cast<size_t>(j) * dc + k], xl[static_cast<size_t>(j) * dc + k]);
      xn[j] += c * c;
      xhn[j] += static_cast<int64_t>(xh[static_cast<size_t>(j) * dc + k]) * xh[static_cast<size_t>(j) * dc + k];
    }
  }
  std::vector<uint8_t> ph(plane_bytes(n, dc)), pl(plane_bytes(n, dc));
  pack_plane(xh.data(), n, dc, ph.data());
  pack_plane(xl.data(), n, dc, pl.data());
  const double step_sq = 1.0 / (65535.0 * 65535.0);
  std::vector<float> full(static_cast<size_t>(m) * n), high(static_cast<size_t>(m) * n);
  for (Engine e : {Engine::kAvx512, Engine::kAmx}) {
    if (e == Engine::kAmx && !amx) continue;
    sq16_l2(e, qh.data(), ql.data(), ph.data(), pl.data(), qn.data(), xn.data(), step_sq, m, n,
            dc, full.data());
    sq16_high_l2(e, qh.data(), ph.data(), qhn.data(), xhn.data(), step_sq, m, n, dc, high.data());
    bool ok_full = true, ok_high = true;
    for (uint32_t i = 0; i < m; ++i) {
      for (uint32_t j = 0; j < n; ++j) {
        int64_t dot = 0, doth = 0;
        for (uint32_t k = 0; k < dc; ++k) {
          dot += code(qh[i * dc + k], ql[i * dc + k]) *
                 code(xh[static_cast<size_t>(j) * dc + k], xl[static_cast<size_t>(j) * dc + k]);
          doth += static_cast<int64_t>(qh[i * dc + k]) * xh[static_cast<size_t>(j) * dc + k];
        }
        const float ref = static_cast<float>(static_cast<double>(qn[i] + xn[j] - 2 * dot) * step_sq);
        const float refh = static_cast<float>(static_cast<double>(qhn[i] + xhn[j] - 2 * doth) * step_sq);
        ok_full &= full[static_cast<size_t>(i) * n + j] == ref;
        ok_high &= high[static_cast<size_t>(i) * n + j] == refh;
      }
    }
    check(ok_full, e == Engine::kAmx ? "amx full" : "avx512 full", m, n, dc);
    check(ok_high, e == Engine::kAmx ? "amx high" : "avx512 high", m, n, dc);
  }
}

void run_bf16(std::mt19937 &rng, uint32_t m, uint32_t n, uint32_t dim, bool amx) {
  std::normal_distribution<float> g(0.0f, 1.0f);
  std::vector<float> q(m * dim), x(static_cast<size_t>(n) * dim);
  for (auto &v : q) v = g(rng);
  for (auto &v : x) v = g(rng);
  std::vector<uint16_t> qb(m * dim), xb(static_cast<size_t>(n) * dim), xp(bf16_packed_elems(n, dim));
  to_bf16(q.data(), qb.data(), qb.size());
  to_bf16(x.data(), xb.data(), xb.size());
  pack_bf16(xb.data(), n, dim, xp.data());
  std::vector<float> qn(m), xn(n);
  for (uint32_t i = 0; i < m; ++i) qn[i] = bf16_norm(qb.data() + i * dim, dim);
  for (uint32_t j = 0; j < n; ++j) xn[j] = bf16_norm(xb.data() + static_cast<size_t>(j) * dim, dim);
  std::vector<float> out(static_cast<size_t>(m) * n);
  for (Engine e : {Engine::kAvx512, Engine::kAmx}) {
    if (e == Engine::kAmx && !amx) continue;
    bf16_l2(e, qb.data(), e == Engine::kAmx ? xp.data() : xb.data(), qn.data(), xn.data(), m, n,
            dim, out.data());
    bool ok = true;
    for (uint32_t i = 0; i < m; ++i) {
      for (uint32_t j = 0; j < n; ++j) {
        double ref = 0;
        for (uint32_t k = 0; k < dim; ++k) {
          auto f = [](uint16_t b) { uint32_t u = static_cast<uint32_t>(b) << 16; float r; std::memcpy(&r, &u, 4); return r; };
          const double dlt = f(qb[i * dim + k]) - f(xb[static_cast<size_t>(j) * dim + k]);
          ref += dlt * dlt;
        }
        ok &= std::fabs(out[static_cast<size_t>(i) * n + j] - ref) <= 1e-3 * (ref + 1.0);
      }
    }
    check(ok, e == Engine::kAmx ? "amx bf16" : "avx512 bf16", m, n, dim);
  }
}

}  // namespace

int main() {
  const bool amx = amx_available();
  std::printf("AMX %s\n", amx ? "available" : "not available, testing AVX-512 only");
  std::mt19937 rng(7);
  for (uint32_t dc : {64u, 128u, 1024u, 3072u})
    for (uint32_t m : {1u, 5u, 16u})
      for (uint32_t n : {1u, 15u, 16u, 17u, 70u, 256u})
        run_case(rng, m, n, dc, amx);
  for (uint32_t dim : {32u, 96u, 1024u})
    for (uint32_t m : {1u, 16u})
      for (uint32_t n : {1u, 17u, 64u})
        run_bf16(rng, m, n, dim, amx);
  std::printf(failures ? "%d failures\n" : "all kernel checks passed\n", failures);
  return failures ? 1 : 0;
}
