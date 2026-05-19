#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>

#include "recsys/ratings_table.hpp"
#include "recsys/similarity.hpp"
#include "recsys/similarity_matrix.hpp"
#include "recsys/thread_pool.hpp"

using recsys::build_item_similarity;
using recsys::build_user_similarity;
using recsys::Metric;
using recsys::Rating;
using recsys::RatingsTable;
using recsys::SimilarityMatrix;
using recsys::SparseVector;
using recsys::ThreadPool;

TEST_CASE("SimilarityMatrix: empty matrix has size 0; size-1 has no pairs") {
    SimilarityMatrix m0;
    CHECK(m0.size() == 0);

    SimilarityMatrix m1(1);
    CHECK(m1.size() == 1);
    CHECK(m1.at(0, 0) == doctest::Approx(1.0f));
}

TEST_CASE("SimilarityMatrix: set/at round-trip is symmetric") {
    SimilarityMatrix m(4);
    m.set(1, 3, 0.42f);
    CHECK(m.at(1, 3) == doctest::Approx(0.42f));
    CHECK(m.at(3, 1) == doctest::Approx(0.42f));
    CHECK(m.at(0, 0) == doctest::Approx(1.0f));
    CHECK(m.at(2, 2) == doctest::Approx(1.0f));
    CHECK(m.at(0, 1) == doctest::Approx(0.0f));  // default
}

TEST_CASE("SimilarityMatrix: out-of-range access throws") {
    SimilarityMatrix m(3);
    CHECK_THROWS_AS(m.at(0, 5), std::out_of_range);
    CHECK_THROWS_AS(m.set(-1, 0, 0.0f), std::out_of_range);
}

TEST_CASE("build_user_similarity (Cosine): every entry matches direct cosine") {
    // 3 users, 4 items.
    std::vector<Rating> raw = {
        {1, 100, 5.0f}, {1, 101, 4.0f}, {1, 102, 3.0f},
        {2, 100, 4.0f}, {2, 102, 2.0f}, {2, 103, 5.0f},
        {3, 101, 3.0f}, {3, 103, 4.0f},
    };
    RatingsTable rt(raw);
    ThreadPool pool(2);
    auto sim = build_user_similarity(rt, Metric::Cosine, pool);

    REQUIRE(sim.size() == rt.num_users());

    for (std::int32_t i = 0; i < rt.num_users(); ++i) {
        for (std::int32_t j = i + 1; j < rt.num_users(); ++j) {
            const auto si = rt.ratings_by_user(i);
            const auto sj = rt.ratings_by_user(j);
            const float direct = recsys::similarity::cosine(
                SparseVector{si.items, si.values, si.count},
                SparseVector{sj.items, sj.values, sj.count});
            CHECK(sim.at(i, j) == doctest::Approx(direct));
            CHECK(sim.at(j, i) == doctest::Approx(direct));  // symmetric
        }
        CHECK(sim.at(i, i) == doctest::Approx(1.0f));
    }
}

TEST_CASE("build_item_similarity (Pearson) and (AdjustedCosine) produce symmetric, in-range values") {
    std::vector<Rating> raw = {
        {1, 100, 5.0f}, {1, 101, 4.0f}, {1, 102, 3.0f},
        {2, 100, 4.0f}, {2, 102, 2.0f}, {2, 103, 5.0f},
        {3, 101, 3.0f}, {3, 103, 4.0f}, {3, 100, 5.0f},
    };
    RatingsTable rt(raw);
    ThreadPool pool(2);

    for (auto metric : {Metric::Pearson, Metric::AdjustedCosine}) {
        auto sim = build_item_similarity(rt, metric, pool);
        REQUIRE(sim.size() == rt.num_items());
        for (std::int32_t i = 0; i < rt.num_items(); ++i) {
            for (std::int32_t j = i + 1; j < rt.num_items(); ++j) {
                const float v = sim.at(i, j);
                CHECK(v == doctest::Approx(sim.at(j, i)));
                CHECK(v >= -1.0001f);
                CHECK(v <= 1.0001f);
            }
        }
    }
}
