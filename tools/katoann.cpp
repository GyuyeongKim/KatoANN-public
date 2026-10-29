// katoann: build an index, measure recall and throughput, or serve an
// open-loop arrival stream and report latency.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "katoann/index.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
  std::map<std::string, std::string> kv;
  std::set<std::string> flags;
  std::string get(const std::string &k, const std::string &def = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? def : it->second;
  }
  std::string need(const std::string &k) const {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::invalid_argument("missing --" + k);
    return it->second;
  }
  uint32_t u32(const std::string &k, uint32_t def) const {
    auto it = kv.find(k);
    return it == kv.end() ? def : static_cast<uint32_t>(std::stoul(it->second));
  }
  double num(const std::string &k, double def) const {
    auto it = kv.find(k);
    return it == kv.end() ? def : std::stod(it->second);
  }
  bool has(const std::string &f) const { return flags.count(f) > 0; }
};

Args parse(int argc, char **argv, int first) {
  static const std::set<std::string> kFlags = {"normalized", "normalize"};
  Args a;
  for (int i = first; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument " + s);
    s = s.substr(2);
    auto eq = s.find('=');
    if (eq != std::string::npos) {
      a.kv[s.substr(0, eq)] = s.substr(eq + 1);
    } else if (kFlags.count(s)) {
      a.flags.insert(s);
    } else {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for --" + s);
      a.kv[s] = argv[++i];
    }
  }
  return a;
}

std::vector<uint32_t> u32_list(const std::string &s) {
  std::vector<uint32_t> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) v.push_back(static_cast<uint32_t>(std::stoul(item)));
  return v;
}

// .fbin: uint32 count, uint32 dim, then float32 rows.
std::vector<float> read_fbin(const std::string &path, uint32_t &n, uint32_t &d) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&d), 4);
  std::vector<float> v(static_cast<size_t>(n) * d);
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float));
  if (!f) throw std::runtime_error("truncated " + path);
  return v;
}

// Ground truth: uint32 count, uint32 k, then count x k uint32 ids.
std::vector<uint32_t> read_gt(const std::string &path, uint32_t &n, uint32_t &k) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&k), 4);
  std::vector<uint32_t> v(static_cast<size_t>(n) * k);
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(uint32_t));
  if (!f) throw std::runtime_error("truncated " + path);
  return v;
}

void normalize_rows(std::vector<float> &v, uint32_t d) {
  for (size_t i = 0; i < v.size(); i += d) {
    double s = 0;
    for (uint32_t k = 0; k < d; ++k) s += static_cast<double>(v[i + k]) * v[i + k];
    if (s > 0) {
      const float inv = static_cast<float>(1.0 / std::sqrt(s));
      for (uint32_t k = 0; k < d; ++k) v[i + k] *= inv;
    }
  }
}

katoann::LoadOptions load_options(const Args &a) {
  katoann::LoadOptions o;
  const std::string prec = a.get("precision", "sq16");
  if (prec == "bf16") o.precision = katoann::Precision::kBf16;
  else if (prec != "sq16") throw std::invalid_argument("--precision is sq16 or bf16");
  const std::string eng = a.get("engine", "amx");
  if (eng == "avx512") o.use_amx = false;
  else if (eng != "amx") throw std::invalid_argument("--engine is amx or avx512");
  o.prefix_dims = a.u32("prefix-dims", 0);
  const std::string coarse = a.get("coarse", "batched");
  if (coarse == "flat") o.batched_coarse = false;
  else if (coarse != "batched") throw std::invalid_argument("--coarse is batched or flat");
  return o;
}

katoann::SearchParams search_params(const Args &a, const katoann::Index &idx) {
  katoann::SearchParams p;
  p.k = a.u32("k", 10);
  const std::string r = a.get("ranking", idx.precision() == katoann::Precision::kBf16 ? "full" : "prefix");
  if (r == "full") p.ranking = katoann::Ranking::kFull;
  else if (r == "high") p.ranking = katoann::Ranking::kHighBytes;
  else if (r == "prefix") p.ranking = katoann::Ranking::kPrefix;
  else throw std::invalid_argument("--ranking is prefix, high, or full");
  p.survivors = a.u32("survivors", 32);
  const std::string s = a.get("selection", "threshold");
  if (s == "local") p.selection = katoann::Selection::kLocal;
  else if (s != "threshold") throw std::invalid_argument("--selection is threshold or local");
  return p;
}

const char *engine_name(katoann::Engine e) {
  return e == katoann::Engine::kAmx ? "amx" : "avx512";
}

double recall_at(const std::vector<uint32_t> &ids, const std::vector<uint32_t> &qid,
                 uint32_t k, const std::vector<uint32_t> &gt, uint32_t gk) {
  double hit = 0;
  for (size_t i = 0; i < qid.size(); ++i) {
    std::set<uint32_t> truth(gt.begin() + static_cast<size_t>(qid[i]) * gk,
                             gt.begin() + static_cast<size_t>(qid[i]) * gk + k);
    for (uint32_t j = 0; j < k; ++j) hit += truth.count(ids[i * k + j]);
  }
  return hit / (static_cast<double>(qid.size()) * k);
}

std::vector<uint32_t> query_ids(const Args &a, uint32_t nq) {
  std::vector<uint32_t> ids;
  const std::string path = a.get("query-ids");
  if (path.empty()) {
    ids.resize(nq);
    for (uint32_t i = 0; i < nq; ++i) ids[i] = i;
    return ids;
  }
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint32_t id;
  while (f >> id) {
    if (id >= nq) throw std::runtime_error("query id out of range in " + path);
    ids.push_back(id);
  }
  if (ids.empty()) throw std::runtime_error("no query ids in " + path);
  return ids;
}

void set_threads(const Args &a) {
  const uint32_t t = a.u32("threads", 0);
  omp_set_dynamic(0);
  if (t) omp_set_num_threads(static_cast<int>(t));
}

int cmd_build(const Args &a) {
  set_threads(a);
  uint32_t n = 0, d = 0;
  std::vector<float> data = read_fbin(a.need("data"), n, d);
  katoann::BuildOptions o;
  o.nlist = a.u32("nlist", 2048);
  o.train_size = a.u32("train-size", 256000);
  o.kmeans_iters = a.u32("iters", 12);
  o.seed = std::stoull(a.get("seed", "0"));
  if (a.has("normalize")) normalize_rows(data, d);
  o.normalized = a.has("normalize") || a.has("normalized");
  const auto t0 = Clock::now();
  katoann::Index::build(data.data(), n, d, o, a.need("out"));
  std::printf("{\"vectors\": %u, \"dim\": %u, \"nlist\": %u, \"build_s\": %.1f}\n", n, d,
              o.nlist, std::chrono::duration<double>(Clock::now() - t0).count());
  return 0;
}

int cmd_search(const Args &a) {
  set_threads(a);
  katoann::Index idx;
  auto t0 = Clock::now();
  idx.load(a.need("index"), load_options(a));
  const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
  uint32_t nq = 0, d = 0;
  std::vector<float> queries = read_fbin(a.need("queries"), nq, d);
  if (d != idx.dim()) throw std::runtime_error("query dimension does not match the index");
  if (idx.normalized()) normalize_rows(queries, d);
  const std::vector<uint32_t> qid = query_ids(a, nq);
  std::vector<float> q(qid.size() * static_cast<size_t>(d));
  for (size_t i = 0; i < qid.size(); ++i)
    std::memcpy(q.data() + i * d, queries.data() + static_cast<size_t>(qid[i]) * d, d * sizeof(float));
  uint32_t gn = 0, gk = 0;
  std::vector<uint32_t> gt;
  if (!a.get("gt").empty()) gt = read_gt(a.get("gt"), gn, gk);
  katoann::SearchParams p = search_params(a, idx);
  if (!gt.empty() && (gn < nq || gk < p.k)) throw std::runtime_error("ground truth too small");
  const uint32_t batch = a.u32("batch", 4096);
  const uint32_t repeat = a.u32("repeat", 3);
  const std::string out = a.get("out");
  std::printf("{\"index\": \"%s\", \"engine\": \"%s\", \"threads\": %d, \"load_s\": %.1f, "
              "\"memory_gb\": %.2f, \"queries\": %zu, \"prefix_dims\": %u}\n",
              a.need("index").c_str(), engine_name(idx.engine()), omp_get_max_threads(),
              load_s, idx.memory_bytes() / 1e9, qid.size(), idx.prefix_dims());
  const size_t total = qid.size();
  std::vector<uint32_t> ids(total * p.k);
  std::vector<float> dists(total * p.k);
  for (uint32_t np : u32_list(a.get("nprobe", "64"))) {
    p.nprobe = np;
    auto pass = [&]() {
      for (size_t off = 0; off < total; off += batch) {
        const uint32_t b = static_cast<uint32_t>(std::min<size_t>(batch, total - off));
        idx.search(q.data() + off * d, b, p, ids.data() + off * p.k, dists.data() + off * p.k);
      }
    };
    pass();  // warmup
    std::vector<double> secs;
    for (uint32_t r = 0; r < repeat; ++r) {
      t0 = Clock::now();
      pass();
      secs.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
    }
    std::sort(secs.begin(), secs.end());
    const double med = secs[secs.size() / 2];
    std::printf("{\"nprobe\": %u, \"k\": %u, \"survivors\": %u, \"batch\": %u, \"qps\": %.1f, "
                "\"seconds_median\": %.4f, \"seconds_min\": %.4f, \"seconds_max\": %.4f",
                np, p.k, p.survivors, batch, total / med, med, secs.front(), secs.back());
    if (!gt.empty()) std::printf(", \"recall\": %.5f", recall_at(ids, qid, p.k, gt, gk));
    std::printf("}\n");
    std::fflush(stdout);
    if (!out.empty()) {
      std::ofstream f(out + ".np" + std::to_string(np) + ".bin", std::ios::binary);
      const uint32_t hdr[2] = {static_cast<uint32_t>(total), p.k};
      f.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
      f.write(reinterpret_cast<const char *>(ids.data()), ids.size() * 4);
      f.write(reinterpret_cast<const char *>(dists.data()), dists.size() * 4);
    }
  }
  return 0;
}

// Requests arrive as a Poisson process. Whenever the index is idle it takes
// every request that has arrived, up to --max-batch, and answers them as one
// batch. Latency runs from arrival to the end of that batch.
int cmd_load(const Args &a) {
  set_threads(a);
  katoann::Index idx;
  idx.load(a.need("index"), load_options(a));
  uint32_t nq = 0, d = 0;
  std::vector<float> queries = read_fbin(a.need("queries"), nq, d);
  if (d != idx.dim()) throw std::runtime_error("query dimension does not match the index");
  if (idx.normalized()) normalize_rows(queries, d);
  const std::vector<uint32_t> pool = query_ids(a, nq);
  katoann::SearchParams p = search_params(a, idx);
  p.nprobe = a.u32("nprobe", 64);
  const double rate = a.num("rate", 1000);
  const uint32_t n = a.u32("requests", 10000);
  const uint32_t max_batch = a.u32("max-batch", 4096);
  std::mt19937_64 rng(a.u32("seed", 42));
  std::exponential_distribution<double> gap(rate);
  std::vector<double> arrival(n);
  std::vector<uint32_t> which(n);
  double t = 0;
  for (uint32_t i = 0; i < n; ++i) {
    t += gap(rng);
    arrival[i] = t;
    which[i] = pool[i % pool.size()];
  }
  std::vector<float> batch_q(static_cast<size_t>(max_batch) * d);
  std::vector<uint32_t> ids(static_cast<size_t>(max_batch) * p.k);
  std::vector<float> dists(static_cast<size_t>(max_batch) * p.k);
  std::vector<double> latency(n);
  std::vector<uint32_t> sizes;
  for (uint32_t w = 0; w < std::min<uint32_t>(max_batch, static_cast<uint32_t>(pool.size())); ++w)
    std::memcpy(batch_q.data() + static_cast<size_t>(w) * d,
                queries.data() + static_cast<size_t>(pool[w]) * d, d * sizeof(float));
  idx.search(batch_q.data(), std::min<uint32_t>(max_batch, static_cast<uint32_t>(pool.size())),
             p, ids.data(), dists.data());  // warmup
  const auto start = Clock::now();
  auto now = [&]() { return std::chrono::duration<double>(Clock::now() - start).count(); };
  uint32_t next = 0;
  while (next < n) {
    double tn = now();
    if (arrival[next] > tn) {
      std::this_thread::sleep_for(std::chrono::duration<double>(arrival[next] - tn));
      tn = now();
    }
    uint32_t end = next;
    while (end < n && end - next < max_batch && arrival[end] <= tn) ++end;
    const uint32_t b = end - next;
    for (uint32_t i = 0; i < b; ++i)
      std::memcpy(batch_q.data() + static_cast<size_t>(i) * d,
                  queries.data() + static_cast<size_t>(which[next + i]) * d, d * sizeof(float));
    idx.search(batch_q.data(), b, p, ids.data(), dists.data());
    const double done = now();
    for (uint32_t i = 0; i < b; ++i) latency[next + i] = done - arrival[next + i];
    sizes.push_back(b);
    next = end;
  }
  const double span = now();
  std::vector<double> sorted = latency;
  std::sort(sorted.begin(), sorted.end());
  auto pct = [&](double q) {
    return sorted[std::min(sorted.size() - 1, static_cast<size_t>(std::ceil(q * sorted.size())) - 1)];
  };
  double mean_batch = 0;
  for (uint32_t s : sizes) mean_batch += s;
  mean_batch /= sizes.size();
  std::printf("{\"rate\": %.1f, \"requests\": %u, \"nprobe\": %u, \"engine\": \"%s\", "
              "\"achieved_qps\": %.1f, \"p50_ms\": %.2f, \"p95_ms\": %.2f, \"p99_ms\": %.2f, "
              "\"max_ms\": %.2f, \"batches\": %zu, \"mean_batch\": %.1f}\n",
              rate, n, p.nprobe, engine_name(idx.engine()), n / span, pct(0.50) * 1e3,
              pct(0.95) * 1e3, pct(0.99) * 1e3, sorted.back() * 1e3, sizes.size(), mean_batch);
  return 0;
}

void usage() {
  std::fprintf(stderr,
      "usage: katoann build  --data BASE.fbin --out PREFIX [--nlist 2048] [--normalize]\n"
      "                      [--train-size 256000] [--iters 12] [--seed 0] [--threads T]\n"
      "       katoann search --index PREFIX --queries Q.fbin [--gt GT.bin] [--k 10]\n"
      "                      [--nprobe 64,128] [--ranking prefix|high|full] [--survivors 32]\n"
      "                      [--prefix-dims D] [--selection threshold|local] [--batch 4096]\n"
      "                      [--precision sq16|bf16] [--engine amx|avx512]\n"
      "                      [--coarse batched|flat] [--repeat 3] [--threads T]\n"
      "                      [--query-ids FILE] [--out PREFIX]\n"
      "       katoann load   --index PREFIX --queries Q.fbin --rate QPS [--requests 10000]\n"
      "                      [--nprobe 64] [--max-batch 4096] [--seed 42] and the search\n"
      "                      options above\n");
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  try {
    const std::string cmd = argv[1];
    const Args a = parse(argc, argv, 2);
    if (cmd == "build") return cmd_build(a);
    if (cmd == "search") return cmd_search(a);
    if (cmd == "load") return cmd_load(a);
    usage();
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "katoann: %s\n", e.what());
    return 1;
  }
}
