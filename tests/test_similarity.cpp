#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "recsys/similarity.hpp"

using recsys::SparseVector;
using recsys::similarity::adjusted_cosine;
using recsys::similarity::cosine;
using recsys::similarity::pearson;

namespace {

SparseVector make_view(const std::vector<std::int32_t>& idx, const std::vector<float>& val) {
    return SparseVector{idx.data(), val.data(), idx.size()};
}

} // namespace

TEST_CASE("cosine: identical vectors give 1.0") {
    std::vector<std::int32_t> idx = {1, 3, 5};
    std::vector<float> val = {1.0f, 2.0f, 3.0f};
    auto v = make_view(idx, val);
    CHECK(cosine(v, v) == doctest::Approx(1.0f));
}

TEST_CASE("cosine: disjoint-index vectors give 0") {
    std::vector<std::int32_t> ai = {1, 2};
    std::vector<float> av = {1.0f, 1.0f};
    std::vector<std::int32_t> bi = {3, 4};
    std::vector<float> bv = {1.0f, 1.0f};
    CHECK(cosine(make_view(ai, av), make_view(bi, bv)) == doctest::Approx(0.0f));
}

TEST_CASE("cosine: known closed-form value on hand-computed example") {
    // a = (1 @ 0, 2 @ 1,        4 @ 3)
    // b = (        2 @ 1, 3 @ 2, 4 @ 3)
    // dot       = 1*0 + 2*2 + 0*3 + 4*4 = 0 + 4 + 0 + 16 = 20
    // |a|^2 = 1 + 4 + 16 = 21
    // |b|^2 = 4 + 9 + 16 = 29
    std::vector<std::int32_t> ai = {0, 1, 3};
    std::vector<float> av = {1.0f, 2.0f, 4.0f};
    std::vector<std::int32_t> bi = {1, 2, 3};
    std::vector<float> bv = {2.0f, 3.0f, 4.0f};
    const float expected = 20.0f / std::sqrt(21.0f * 29.0f);
    CHECK(cosine(make_view(ai, av), make_view(bi, bv)) == doctest::Approx(expected));
}

TEST_CASE("cosine: unsorted indices produce the same result as sorted") {
    std::vector<std::int32_t> sorted_idx = {0, 1, 2, 3};
    std::vector<float> sorted_val = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::int32_t> shuffled_idx = {3, 0, 2, 1};
    std::vector<float> shuffled_val = {4.0f, 1.0f, 3.0f, 2.0f};

    auto a_sorted = make_view(sorted_idx, sorted_val);
    auto a_shuf = make_view(shuffled_idx, shuffled_val);

    std::vector<std::int32_t> bi = {1, 2, 3};
    std::vector<float> bv = {1.0f, 1.0f, 1.0f};
    auto b = make_view(bi, bv);

    CHECK(cosine(a_sorted, b) == doctest::Approx(cosine(a_shuf, b)));
}

TEST_CASE("cosine: empty or zero-norm input yields 0") {
    SparseVector empty{nullptr, nullptr, 0};
    std::vector<std::int32_t> idx = {1};
    std::vector<float> val = {2.0f};
    auto v = make_view(idx, val);
    CHECK(cosine(empty, v) == doctest::Approx(0.0f));
    CHECK(cosine(v, empty) == doctest::Approx(0.0f));

    std::vector<float> zeros = {0.0f};
    auto zero_v = make_view(idx, zeros);
    CHECK(cosine(zero_v, v) == doctest::Approx(0.0f));
}

TEST_CASE("pearson: perfectly co-varying vectors give 1.0") {
    // Over the overlap (indices 0,1,2), b = 2*a, so they co-vary perfectly.
    std::vector<std::int32_t> ai = {0, 1, 2};
    std::vector<float> av = {1.0f, 2.0f, 3.0f};
    std::vector<std::int32_t> bi = {0, 1, 2};
    std::vector<float> bv = {2.0f, 4.0f, 6.0f};
    const float mean_a = 2.0f;
    const float mean_b = 4.0f;
    CHECK(pearson(make_view(ai, av), mean_a, make_view(bi, bv), mean_b) == doctest::Approx(1.0f));
}

TEST_CASE("pearson: perfectly anti-correlated vectors give -1.0") {
    std::vector<std::int32_t> ai = {0, 1, 2};
    std::vector<float> av = {1.0f, 2.0f, 3.0f};
    std::vector<std::int32_t> bi = {0, 1, 2};
    std::vector<float> bv = {3.0f, 2.0f, 1.0f};
    CHECK(pearson(make_view(ai, av), 2.0f, make_view(bi, bv), 2.0f) == doctest::Approx(-1.0f));
}

TEST_CASE("pearson: empty intersection or constant vector yields 0") {
    std::vector<std::int32_t> ai = {0, 1};
    std::vector<float> av = {1.0f, 2.0f};
    std::vector<std::int32_t> bi = {5, 6};
    std::vector<float> bv = {3.0f, 4.0f};
    CHECK(pearson(make_view(ai, av), 1.5f, make_view(bi, bv), 3.5f) == doctest::Approx(0.0f));

    // Same index, but a is constant on the intersection -> sum_sq = 0.
    std::vector<std::int32_t> ci = {0, 1};
    std::vector<float> cv = {2.0f, 2.0f};
    CHECK(pearson(make_view(ci, cv), 2.0f, make_view(ai, av), 1.5f) == doctest::Approx(0.0f));
}

TEST_CASE("adjusted_cosine: centers by per-index mean (user-mean for items)") {
    // Two item vectors indexed by user. Users 0..2 have means stored at
    // mean_per_index[0..2]. After subtracting user means, the items are
    // perfectly co-varying -> sim = 1.0.
    std::vector<float> user_means = {3.0f, 2.0f, 4.0f};
    std::vector<std::int32_t> ai = {0, 1, 2};
    std::vector<float> av = {4.0f, 3.0f, 5.0f}; // centered: +1, +1, +1
    std::vector<std::int32_t> bi = {0, 1, 2};
    std::vector<float> bv = {5.0f, 4.0f, 6.0f}; // centered: +2, +2, +2
    CHECK(adjusted_cosine(make_view(ai, av), make_view(bi, bv), user_means.data()) ==
          doctest::Approx(1.0f));

    // Flip the sign of b's deviations -> anti-correlation.
    std::vector<float> bv_anti = {1.0f, 0.0f, 2.0f}; // centered: -2, -2, -2
    CHECK(adjusted_cosine(make_view(ai, av), make_view(bi, bv_anti), user_means.data()) ==
          doctest::Approx(-1.0f));
}
