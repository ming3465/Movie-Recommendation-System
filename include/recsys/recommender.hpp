#pragma once

#include <cstdint>

namespace recsys {

class RatingsTable;
class SimilarityMatrix;

// User-based k-NN with mean-centered weighted-average prediction:
//
//     predicted(u, i) = mean(u) + sum_{v in top-k neighbors who rated i}
//                                   sim(u, v) * (r(v, i) - mean(v))
//                                 / sum_{v in top-k}  |sim(u, v)|
//
// "Top-k" is ranked by sim(u, v) descending (signed), among users who have
// rated item i and have non-zero similarity to u. Falls back to mean(u) when
// no usable neighbor exists.
class UserKNN {
public:
    UserKNN(const RatingsTable& rt, const SimilarityMatrix& user_sim, int k);

    float predict(std::int32_t user_idx, std::int32_t item_idx) const;

private:
    const RatingsTable& rt_;
    const SimilarityMatrix& sim_;
    int k_;
};

}  // namespace recsys
