#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>

#include "recsys/evaluator.hpp"

using recsys::Evaluator;

TEST_CASE("Evaluator: empty accumulator reports zeros") {
    Evaluator e;
    const auto m = e.result();
    CHECK(m.scored == 0);
    CHECK(m.rmse == doctest::Approx(0.0));
    CHECK(m.mae == doctest::Approx(0.0));
}

TEST_CASE("Evaluator: perfect predictions give RMSE=MAE=0") {
    Evaluator e;
    e.add(3.5f, 3.5f);
    e.add(1.0f, 1.0f);
    e.add(5.0f, 5.0f);
    const auto m = e.result();
    CHECK(m.scored == 3);
    CHECK(m.rmse == doctest::Approx(0.0));
    CHECK(m.mae == doctest::Approx(0.0));
}

TEST_CASE("Evaluator: closed-form RMSE/MAE on a hand-computed example") {
    // predicted: 1, 2, 3   actual: 1, 2, 4  -> errors: 0, 0, -1
    // MAE  = 1/3
    // RMSE = sqrt(1/3)
    Evaluator e;
    e.add(1.0f, 1.0f);
    e.add(2.0f, 2.0f);
    e.add(3.0f, 4.0f);
    const auto m = e.result();
    CHECK(m.scored == 3);
    CHECK(m.mae == doctest::Approx(1.0 / 3.0));
    CHECK(m.rmse == doctest::Approx(std::sqrt(1.0 / 3.0)));
}
