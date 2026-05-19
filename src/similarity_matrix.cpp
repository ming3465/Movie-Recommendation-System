#include "recsys/similarity_matrix.hpp"

#include <stdexcept>

#include "recsys/thread_pool.hpp"

namespace recsys {

namespace {

SparseVector user_view(const RatingsTable::UserSlice& s) {
    return SparseVector{s.items, s.values, s.count};
}
SparseVector item_view(const RatingsTable::ItemSlice& s) {
    return SparseVector{s.users, s.values, s.count};
}

float user_pair_similarity(const RatingsTable& rt,
                           std::int32_t i, std::int32_t j, Metric m) {
    const auto a = user_view(rt.ratings_by_user(i));
    const auto b = user_view(rt.ratings_by_user(j));
    switch (m) {
        case Metric::Cosine:
            return similarity::cosine(a, b);
        case Metric::Pearson:
            return similarity::pearson(a, rt.user_mean(i), b, rt.user_mean(j));
        case Metric::AdjustedCosine:
            // User-vectors are indexed by item; center by item-mean.
            return similarity::adjusted_cosine(a, b, rt.item_means_data());
    }
    return 0.0f;
}

float item_pair_similarity(const RatingsTable& rt,
                           std::int32_t i, std::int32_t j, Metric m) {
    const auto a = item_view(rt.ratings_by_item(i));
    const auto b = item_view(rt.ratings_by_item(j));
    switch (m) {
        case Metric::Cosine:
            return similarity::cosine(a, b);
        case Metric::Pearson:
            return similarity::pearson(a, rt.item_mean(i), b, rt.item_mean(j));
        case Metric::AdjustedCosine:
            // Item-vectors are indexed by user; center by user-mean.
            return similarity::adjusted_cosine(a, b, rt.user_means_data());
    }
    return 0.0f;
}

SimilarityMatrix build_similarity(
    std::int32_t n,
    ThreadPool& pool,
    float (*pair_fn)(const RatingsTable&, std::int32_t, std::int32_t, Metric),
    const RatingsTable& rt,
    Metric m) {
    SimilarityMatrix out(n);
    if (n < 2) return out;
    pool.parallel_for(static_cast<std::size_t>(n),
                      [&](std::size_t b, std::size_t e) {
                          for (std::size_t i = b; i < e; ++i) {
                              for (std::size_t j = i + 1;
                                   j < static_cast<std::size_t>(n); ++j) {
                                  const float s = pair_fn(rt,
                                      static_cast<std::int32_t>(i),
                                      static_cast<std::int32_t>(j), m);
                                  out.set(static_cast<std::int32_t>(i),
                                          static_cast<std::int32_t>(j), s);
                              }
                          }
                      });
    return out;
}

}  // namespace

SimilarityMatrix::SimilarityMatrix(std::int32_t n) : n_(n) {
    if (n < 0) throw std::invalid_argument("SimilarityMatrix: negative size");
    const std::size_t slots =
        (n_ < 2) ? 0 : static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_ - 1) / 2;
    upper_.assign(slots, 0.0f);
}

std::size_t SimilarityMatrix::flat_index(std::int32_t i, std::int32_t j,
                                         std::int32_t n) {
    const auto in = static_cast<std::size_t>(n);
    const auto ii = static_cast<std::size_t>(i);
    const auto jj = static_cast<std::size_t>(j);
    return ii * (2 * in - ii - 1) / 2 + (jj - ii - 1);
}

float SimilarityMatrix::at(std::int32_t i, std::int32_t j) const {
    if (i < 0 || j < 0 || i >= n_ || j >= n_) {
        throw std::out_of_range("SimilarityMatrix::at: index out of range");
    }
    if (i == j) return 1.0f;
    if (i > j) std::swap(i, j);
    return upper_[flat_index(i, j, n_)];
}

void SimilarityMatrix::set(std::int32_t i, std::int32_t j, float value) {
    if (i < 0 || j < 0 || i >= n_ || j >= n_) {
        throw std::out_of_range("SimilarityMatrix::set: index out of range");
    }
    if (i == j) return;
    if (i > j) std::swap(i, j);
    upper_[flat_index(i, j, n_)] = value;
}

SimilarityMatrix build_user_similarity(const RatingsTable& rt,
                                       Metric metric, ThreadPool& pool) {
    return build_similarity(rt.num_users(), pool, user_pair_similarity, rt, metric);
}

SimilarityMatrix build_item_similarity(const RatingsTable& rt,
                                       Metric metric, ThreadPool& pool) {
    return build_similarity(rt.num_items(), pool, item_pair_similarity, rt, metric);
}

}  // namespace recsys
