#include "recsys/evaluator.hpp"

#include <cmath>

namespace recsys {

void Evaluator::add(float predicted, float actual) {
    const double err = static_cast<double>(predicted) - static_cast<double>(actual);
    sum_abs_ += std::fabs(err);
    sum_sq_ += err * err;
    ++n_;
}

EvalMetrics Evaluator::result() const {
    EvalMetrics m;
    m.scored = n_;
    if (n_ == 0) return m;
    m.mae = sum_abs_ / static_cast<double>(n_);
    m.rmse = std::sqrt(sum_sq_ / static_cast<double>(n_));
    return m;
}

}  // namespace recsys
