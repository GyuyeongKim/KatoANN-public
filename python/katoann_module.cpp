#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "katoann/index.h"

namespace py = pybind11;
using katoann::Index;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

katoann::Ranking parse_ranking(const std::string &s) {
  if (s == "prefix") return katoann::Ranking::kPrefix;
  if (s == "high") return katoann::Ranking::kHighBytes;
  if (s == "full") return katoann::Ranking::kFull;
  throw std::invalid_argument("ranking is 'prefix', 'high' or 'full'");
}

katoann::Selection parse_selection(const std::string &s) {
  if (s == "threshold") return katoann::Selection::kThreshold;
  if (s == "local") return katoann::Selection::kLocal;
  throw std::invalid_argument("selection is 'threshold' or 'local'");
}

}  // namespace

PYBIND11_MODULE(katoann, m) {
  m.doc() = "KatoANN: IVF vector search with 16-bit codes on CPU matrix engines";
  m.def("amx_available", &katoann::amx_available,
        "True if the CPU has AMX and Linux grants this process the tile state");

  m.def(
      "build",
      [](FloatArray data, const std::string &prefix, uint32_t nlist, uint32_t train_size,
         uint32_t kmeans_iters, uint64_t seed, bool normalized) {
        if (data.ndim() != 2) throw std::invalid_argument("data must be a 2-D float32 array");
        katoann::BuildOptions o;
        o.nlist = nlist;
        o.train_size = train_size;
        o.kmeans_iters = kmeans_iters;
        o.seed = seed;
        o.normalized = normalized;
        const float *p = data.data();
        const auto n = static_cast<uint32_t>(data.shape(0));
        const auto d = static_cast<uint32_t>(data.shape(1));
        py::gil_scoped_release release;
        Index::build(p, n, d, o, prefix);
      },
      py::arg("data"), py::arg("prefix"), py::arg("nlist") = 2048, py::arg("train_size") = 256000,
      py::arg("kmeans_iters") = 12, py::arg("seed") = 0, py::arg("normalized") = false,
      "Partition data (n x dim float32) and write the index files under prefix");

  py::class_<Index>(m, "Index")
      .def(py::init<>())
      .def(
          "load",
          [](Index &self, const std::string &prefix, const std::string &precision, bool use_amx,
             uint32_t prefix_dims, bool batched_coarse) {
            katoann::LoadOptions o;
            if (precision == "bf16") o.precision = katoann::Precision::kBf16;
            else if (precision != "sq16") throw std::invalid_argument("precision is 'sq16' or 'bf16'");
            o.use_amx = use_amx;
            o.prefix_dims = prefix_dims;
            o.batched_coarse = batched_coarse;
            py::gil_scoped_release release;
            self.load(prefix, o);
          },
          py::arg("prefix"), py::arg("precision") = "sq16", py::arg("use_amx") = true,
          py::arg("prefix_dims") = 0, py::arg("batched_coarse") = true)
      .def(
          "search",
          [](const Index &self, FloatArray queries, uint32_t k, uint32_t nprobe,
             const std::string &ranking, uint32_t survivors, const std::string &selection) {
            if (queries.ndim() != 2 || static_cast<uint32_t>(queries.shape(1)) != self.dim())
              throw std::invalid_argument("queries must be (n, dim) float32");
            katoann::SearchParams p;
            p.k = k;
            p.nprobe = nprobe;
            p.ranking = parse_ranking(ranking);
            p.survivors = survivors;
            p.selection = parse_selection(selection);
            const auto nq = static_cast<uint32_t>(queries.shape(0));
            py::array_t<uint32_t> ids({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
            py::array_t<float> dists({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
            const float *q = queries.data();
            uint32_t *pi = ids.mutable_data();
            float *pd = dists.mutable_data();
            {
              py::gil_scoped_release release;
              std::vector<float> unit;
              if (self.normalized()) {
                const uint32_t d = self.dim();
                unit.assign(q, q + static_cast<size_t>(nq) * d);
                for (size_t i = 0; i < unit.size(); i += d) {
                  double s = 0;
                  for (uint32_t k = 0; k < d; ++k) s += static_cast<double>(unit[i + k]) * unit[i + k];
                  if (s > 0) {
                    const float inv = static_cast<float>(1.0 / std::sqrt(s));
                    for (uint32_t k = 0; k < d; ++k) unit[i + k] *= inv;
                  }
                }
                q = unit.data();
              }
              self.search(q, nq, p, pi, pd);
            }
            return py::make_tuple(ids, dists);
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("nprobe") = 64,
          py::arg("ranking") = "prefix", py::arg("survivors") = 32,
          py::arg("selection") = "threshold",
          "Search all queries together; returns (ids, squared L2 distances), nearest first. "
          "Queries are normalized when the index was built with normalized=True")
      .def_property_readonly("dim", &Index::dim)
      .def_property_readonly("nlist", &Index::nlist)
      .def_property_readonly("size", &Index::size)
      .def_property_readonly("normalized", &Index::normalized)
      .def_property_readonly("prefix_dims", &Index::prefix_dims)
      .def_property_readonly("engine", [](const Index &self) {
        return self.engine() == katoann::Engine::kAmx ? "amx" : "avx512";
      })
      .def_property_readonly("memory_bytes", &Index::memory_bytes);
}
