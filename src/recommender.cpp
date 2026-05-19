#include "recsys/recommender.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "recsys/ratings_table.hpp"
#include "recsys/similarity_matrix.hpp"

namespace recsys {

namespace {

struct Neighbor {
    float sim;
    float deviation;  // rating - neighbor's mean
};

}  // namespace

UserKNN::UserKNN(const RatingsTable& rt, const SimilarityMatrix& user_sim, int k)
    : rt_(rt), sim_(user_sim), k_(k) {}

float UserKNN::predict(std::int32_t user_idx, std::int32_t item_idx) const {
    const float user_mean = rt_.user_mean(user_idx);

    const auto raters = rt_.ratings_by_item(item_idx);
    if (raters.count == 0 || k_ <= 0) return user_mean;

    std::vector<Neighbor> cand;
    cand.reserve(raters.count);
    for (std::size_t i = 0; i < raters.count; ++i) {
        const std::int32_t v = raters.users[i];
        if (v == user_idx) continue;
        const float s = sim_.at(user_idx, v);
        if (s == 0.0f) continue;
        cand.push_back({s, raters.values[i] - rt_.user_mean(v)});
    }
    if (cand.empty()) return user_mean;

    const std::size_t k = std::min(static_cast<std::size_t>(k_), cand.size());
    std::partial_sort(
        cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(k), cand.end(),
        [](const Neighbor& a, const Neighbor& b) { return a.sim > b.sim; });

    float num = 0.0f, denom = 0.0f;
    for (std::size_t i = 0; i < k; ++i) {
        num += cand[i].sim * cand[i].deviation;
        denom += std::fabs(cand[i].sim);
    }
    if (denom <= 0.0f) return user_mean;
    return user_mean + num / denom;
}

}  // namespace recsys
