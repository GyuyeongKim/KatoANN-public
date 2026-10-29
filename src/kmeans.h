#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// k-means for the coarse partition. Adapted from PipeANN (Apache-2.0), which
// derives from DiskANN (MIT); see NOTICE.
namespace katoann::kmeans {

float l2sq(const float *a, const float *b, uint32_t dim);

// k nearest centers of every point, num_points x k, nearest first.
void closest_centers(const float *data, size_t num_points, size_t dim,
                     const float *centers, size_t num_centers, size_t k,
                     uint32_t *out);

// k-means++ seeding. seed 0 draws a seed from std::random_device.
void kmeanspp(const float *data, size_t num_points, size_t dim,
              float *centers, size_t num_centers, uint64_t seed);

// Elkan's k-means for at most max_iters iterations; centers are updated in place.
float elkan(const float *data, size_t num_points, size_t dim, float *centers,
            size_t num_centers, size_t max_iters);

}  // namespace katoann::kmeans
