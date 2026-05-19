#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>

#include "recsys/ratings_table.hpp"
#include "recsys/recommender.hpp"
#include "recsys/similarity_matrix.hpp"

using recsys::Rating;
using recsys::RatingsTable;
using recsys::SimilarityMatrix;
using recsys::UserKNN;

TEST_CASE("UserKNN.predict: mean-centered weighted average on a hand-computed case") {
    // user A (idx 0): item 100 -> 5, item 101 -> 4; mean = 4.5
    // user B (idx 1): item 100 -> 4, item 101 -> 3; mean = 3.5
    // user C (idx 2): item 100 -> 2;                mean = 2.0
    std::vector<Rating> raw = {
        {1, 100, 5.0f}, {1, 101, 4.0f},
        {2, 100, 4.0f}, {2, 101, 3.0f},
        {3, 100, 2.0f},
    };
    RatingsTable rt(raw);
    REQUIRE(rt.num_users() == 3);
    REQUIRE(rt.num_items() == 2);

    // Hand-set similarity. C is equally similar to A and B.
    SimilarityMatrix sim(3);
    sim.set(2, 0, 1.0f);
    sim.set(2, 1, 1.0f);
    sim.set(0, 1, 0.5f);

    UserKNN model(rt, sim, /*k=*/2);
    // C predicts item 101 (internal idx 1):
    //   deviations: A: 4 - 4.5 = -0.5; B: 3 - 3.5 = -0.5
    //   predicted = 2.0 + (1*(-0.5) + 1*(-0.5)) / (1 + 1) = 1.5
    CHECK(model.predict(/*user=*/2, /*item=*/1) == doctest::Approx(1.5f));
}

TEST_CASE("UserKNN.predict: weights asymmetric neighbors correctly") {
    // Same setup, but make A much more similar to C than B is.
    std::vector<Rating> raw = {
        {1, 100, 5.0f}, {1, 101, 4.0f},
        {2, 100, 4.0f}, {2, 101, 2.0f},
        {3, 100, 2.0f},
    };
    RatingsTable rt(raw);
    SimilarityMatrix sim(3);
    sim.set(2, 0, 0.9f);  // sim(C, A)
    sim.set(2, 1, 0.1f);  // sim(C, B)

    UserKNN model(rt, sim, 2);
    // user_mean(C) = 2.0
    // A: rating 4, mean 4.5, dev = -0.5
    // B: rating 2, mean 3.0, dev = -1.0
    // predicted = 2.0 + (0.9*-0.5 + 0.1*-1.0) / (0.9 + 0.1)
    //           = 2.0 + (-0.45 + -0.10) / 1.0
    //           = 2.0 - 0.55 = 1.45
    CHECK(model.predict(2, 1) == doctest::Approx(1.45f));
}

TEST_CASE("UserKNN.predict: cold item (nobody rated it) returns user mean") {
    // Build with 2 items, then ask about item index 1 in a fresh table
    // where only item 0 has a rater.
    std::vector<Rating> raw = {
        {1, 100, 5.0f}, {1, 101, 4.0f},  // user 0 rates both
        {2, 100, 4.0f},                  // user 1 rates only item 100
    };
    RatingsTable rt(raw);
    SimilarityMatrix sim(rt.num_users());

    UserKNN model(rt, sim, 5);
    // Construct a fake "cold item" scenario by predicting for a real item
    // but with all sims = 0 (no usable neighbor).
    // user 1 mean = 4.0; predict item 1 (which user 1 didn't rate):
    // The only rater of item 1 is user 0, but sim(1, 0) = 0 -> fallback.
    CHECK(model.predict(1, 1) == doctest::Approx(rt.user_mean(1)));
}

TEST_CASE("UserKNN.predict: k=0 always returns user mean") {
    std::vector<Rating> raw = {
        {1, 100, 5.0f}, {1, 101, 4.0f},
        {2, 100, 4.0f}, {2, 101, 3.0f},
    };
    RatingsTable rt(raw);
    SimilarityMatrix sim(rt.num_users());
    sim.set(0, 1, 1.0f);
    UserKNN model(rt, sim, /*k=*/0);
    CHECK(model.predict(0, 1) == doctest::Approx(rt.user_mean(0)));
}
