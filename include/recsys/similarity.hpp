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

namespace similarity {

// Cosine similarity in [-1, 1]. Returns 0 if either vector is empty or has
// zero L2 norm.
float cosine(const SparseVector& a, const SparseVector& b);

}  // namespace similarity
}  // namespace recsys
