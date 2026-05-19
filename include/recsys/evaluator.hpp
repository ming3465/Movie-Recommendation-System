#pragma once

#include <cstddef>

namespace recsys {

struct EvalMetrics {
    double rmse = 0.0;
    double mae = 0.0;
    std::size_t scored = 0;
};

// Streaming accumulator for prediction-quality metrics. Call add() once per
// (predicted, actual) pair; call result() to read RMSE / MAE / count.
class Evaluator {
public:
    void add(float predicted, float actual);
    EvalMetrics result() const;

private:
    double sum_abs_ = 0.0;
    double sum_sq_ = 0.0;
    std::size_t n_ = 0;
};

}  // namespace recsys
