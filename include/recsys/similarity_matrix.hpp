#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "recsys/ratings_table.hpp"
#include "recsys/similarity.hpp"

namespace recsys {

class ThreadPool;

// Symmetric similarity matrix stored as a flat upper-triangle buffer of
// length n*(n-1)/2. `at(i, j)` returns the same value for any order, and
// returns 1 on the diagonal.
class SimilarityMatrix {
public:
    SimilarityMatrix() = default;
    explicit SimilarityMatrix(std::int32_t n);

    std::int32_t size() const noexcept {
        return n_;
    }

    float at(std::int32_t i, std::int32_t j) const;
    void set(std::int32_t i, std::int32_t j, float value);

private:
    static std::size_t flat_index(std::int32_t i, std::int32_t j, std::int32_t n);

    std::int32_t n_ = 0;
    std::vector<float> upper_;
};

// Build the user-user similarity matrix for `rt` using `metric`. Per-pair
// work is dispatched across `pool` via parallel_for.
SimilarityMatrix build_user_similarity(const RatingsTable& rt, Metric metric, ThreadPool& pool);

// Same, item-item.
SimilarityMatrix build_item_similarity(const RatingsTable& rt, Metric metric, ThreadPool& pool);

} // namespace recsys
