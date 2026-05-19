#pragma once

#include <cstddef>
#include <cstdint>

namespace recsys {

// Non-owning view of a sparse vector. `indices` and `values` are parallel
// arrays of length `count`. The index order is not required to be sorted;
// the similarity functions handle arbitrary input order.
struct SparseVector {
    const std::int32_t* indices;
    const float* values;
    std::size_t count;
};

enum class Metric {
    Cosine,
    Pearson,
    AdjustedCosine,
};

namespace similarity {

// Cosine similarity in [-1, 1]. Returns 0 if either vector is empty or has
// zero L2 norm.
float cosine(const SparseVector& a, const SparseVector& b);

// Pearson correlation computed over the intersection of `a`'s and `b`'s
// indices. The two means must be the vectors' means across their full
// rating sets (typically `user_mean` from RatingsTable). Returns 0 if the
// intersection is empty or either centered vector is constant.
float pearson(const SparseVector& a, float mean_a,
              const SparseVector& b, float mean_b);

// Adjusted cosine: same form as Pearson but the value subtracted at each
// position is `mean_per_index[idx]` rather than a single per-vector mean.
// Used for item-item similarity, where item vectors are indexed by user
// and the centering is the user's mean. `mean_per_index` must be large
// enough to cover every index appearing in `a` or `b`.
float adjusted_cosine(const SparseVector& a, const SparseVector& b,
                      const float* mean_per_index);

}  // namespace similarity
}  // namespace recsys
