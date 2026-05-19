#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <vector>

#include "recsys/ratings_table.hpp"

using recsys::Rating;
using recsys::RatingsTable;
using recsys::Split;
using recsys::train_test_split;

TEST_CASE("RatingsTable: counts and means on a 4x3 toy matrix") {
    // Users 10, 20, 30, 40; items 1, 2, 3.
    std::vector<Rating> raw = {
        {10, 1, 5.0f}, {10, 2, 3.0f},
        {20, 1, 4.0f},
        {30, 2, 2.0f}, {30, 3, 1.0f},
        {40, 3, 5.0f},
    };
    RatingsTable t(raw);

    CHECK(t.num_users() == 4);
    CHECK(t.num_items() == 3);
    CHECK(t.num_ratings() == 6);

    // Internal indices are assigned in order of first appearance.
    CHECK(t.user_index(10) == 0);
    CHECK(t.user_index(40) == 3);
    CHECK(t.user_index(999) == -1);
    CHECK(t.item_index(1) == 0);
    CHECK(t.item_index(3) == 2);

    CHECK(t.user_mean(0) == doctest::Approx(4.0f));  // (5+3)/2
    CHECK(t.user_mean(1) == doctest::Approx(4.0f));  // 4
    CHECK(t.user_mean(2) == doctest::Approx(1.5f));  // (2+1)/2
    CHECK(t.user_mean(3) == doctest::Approx(5.0f));  // 5

    CHECK(t.item_mean(0) == doctest::Approx(4.5f));  // (5+4)/2
    CHECK(t.item_mean(1) == doctest::Approx(2.5f));  // (3+2)/2
    CHECK(t.item_mean(2) == doctest::Approx(3.0f));  // (1+5)/2

    CHECK(t.global_mean() == doctest::Approx(20.0f / 6.0f));
}

TEST_CASE("RatingsTable: CSR slices expose every rating exactly once") {
    std::vector<Rating> raw = {
        {10, 1, 5.0f}, {10, 2, 3.0f},
        {20, 1, 4.0f},
    };
    RatingsTable t(raw);

    auto u0 = t.ratings_by_user(0);
    REQUIRE(u0.count == 2);
    std::vector<std::int32_t> items(u0.items, u0.items + u0.count);
    std::sort(items.begin(), items.end());
    CHECK(items[0] == 0);
    CHECK(items[1] == 1);

    auto i0 = t.ratings_by_item(0);
    REQUIRE(i0.count == 2);
    float sum = 0.0f;
    for (std::size_t k = 0; k < i0.count; ++k) sum += i0.values[k];
    CHECK(sum == doctest::Approx(9.0f));  // 5 + 4
}

TEST_CASE("RatingsTable: out-of-range slice access throws") {
    RatingsTable t(std::vector<Rating>{{10, 1, 5.0f}});
    CHECK_THROWS_AS(t.ratings_by_user(99), std::out_of_range);
    CHECK_THROWS_AS(t.ratings_by_item(99), std::out_of_range);
    CHECK_THROWS_AS(t.ratings_by_user(-1), std::out_of_range);
}

TEST_CASE("train_test_split: partitions all ratings, fraction in window, reproducible") {
    std::vector<Rating> raw;
    raw.reserve(5000);
    for (int u = 0; u < 100; ++u) {
        for (int i = 0; i < 50; ++i) {
            raw.push_back({u, i, 1.0f});
        }
    }
    REQUIRE(raw.size() == 5000);

    const auto s = train_test_split(raw, 0.2f, 42);
    CHECK(s.train.size() + s.test.size() == raw.size());

    const float test_frac =
        static_cast<float>(s.test.size()) / static_cast<float>(raw.size());
    CHECK(test_frac > 0.17f);
    CHECK(test_frac < 0.23f);

    // Same seed -> identical split.
    const auto s2 = train_test_split(raw, 0.2f, 42);
    REQUIRE(s2.test.size() == s.test.size());
    for (std::size_t k = 0; k < s.test.size(); ++k) {
        CHECK(s.test[k].user_id == s2.test[k].user_id);
        CHECK(s.test[k].item_id == s2.test[k].item_id);
    }

    // Different seed -> different split (probabilistic; almost-certainly).
    const auto s3 = train_test_split(raw, 0.2f, 7);
    bool any_different = false;
    const auto n = std::min(s.test.size(), s3.test.size());
    for (std::size_t k = 0; k < n; ++k) {
        if (s.test[k].user_id != s3.test[k].user_id ||
            s.test[k].item_id != s3.test[k].item_id) {
            any_different = true;
            break;
        }
    }
    CHECK(any_different);
}

TEST_CASE("train_test_split: rejects fractions outside [0, 1]") {
    std::vector<Rating> raw = {{0, 0, 1.0f}};
    CHECK_THROWS_AS(train_test_split(raw, -0.1f, 1), std::invalid_argument);
    CHECK_THROWS_AS(train_test_split(raw, 1.5f, 1), std::invalid_argument);
}
