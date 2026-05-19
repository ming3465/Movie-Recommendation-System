#include "recsys/similarity.hpp"

#include <cmath>
#include <unordered_map>

namespace recsys::similarity {

float cosine(const SparseVector& a, const SparseVector& b) {
    if (a.count == 0 || b.count == 0)
        return 0.0f;

    float norm_a_sq = 0.0f;
    for (std::size_t i = 0; i < a.count; ++i) {
        norm_a_sq += a.values[i] * a.values[i];
    }
    float norm_b_sq = 0.0f;
    for (std::size_t i = 0; i < b.count; ++i) {
        norm_b_sq += b.values[i] * b.values[i];
    }
    if (norm_a_sq <= 0.0f || norm_b_sq <= 0.0f)
        return 0.0f;

    // Dot product on (possibly unsorted) sparse indices: hash the shorter
    // vector, then probe with entries from the longer one.
    const SparseVector& shorter = (a.count <= b.count) ? a : b;
    const SparseVector& longer = (a.count <= b.count) ? b : a;

    std::unordered_map<std::int32_t, float> idx_to_val;
    idx_to_val.reserve(shorter.count);
    for (std::size_t i = 0; i < shorter.count; ++i) {
        idx_to_val.emplace(shorter.indices[i], shorter.values[i]);
    }

    float dot = 0.0f;
    for (std::size_t i = 0; i < longer.count; ++i) {
        const auto it = idx_to_val.find(longer.indices[i]);
        if (it != idx_to_val.end()) {
            dot += longer.values[i] * it->second;
        }
    }

    return dot / (std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq));
}

namespace {

// Build a quick lookup from one vector's (index -> value) pairs.
std::unordered_map<std::int32_t, float> index_map(const SparseVector& v) {
    std::unordered_map<std::int32_t, float> m;
    m.reserve(v.count);
    for (std::size_t i = 0; i < v.count; ++i) {
        m.emplace(v.indices[i], v.values[i]);
    }
    return m;
}

} // namespace

float pearson(const SparseVector& a, float mean_a, const SparseVector& b, float mean_b) {
    if (a.count == 0 || b.count == 0)
        return 0.0f;

    const SparseVector& shorter = (a.count <= b.count) ? a : b;
    const SparseVector& longer = (a.count <= b.count) ? b : a;
    const float mean_s = (a.count <= b.count) ? mean_a : mean_b;
    const float mean_l = (a.count <= b.count) ? mean_b : mean_a;

    const auto lookup = index_map(shorter);

    float num = 0.0f, sum_sq_s = 0.0f, sum_sq_l = 0.0f;
    for (std::size_t i = 0; i < longer.count; ++i) {
        const auto it = lookup.find(longer.indices[i]);
        if (it == lookup.end())
            continue;
        const float ds = it->second - mean_s;
        const float dl = longer.values[i] - mean_l;
        num += ds * dl;
        sum_sq_s += ds * ds;
        sum_sq_l += dl * dl;
    }
    if (sum_sq_s <= 0.0f || sum_sq_l <= 0.0f)
        return 0.0f;
    return num / std::sqrt(sum_sq_s * sum_sq_l);
}

float adjusted_cosine(const SparseVector& a, const SparseVector& b, const float* mean_per_index) {
    if (a.count == 0 || b.count == 0)
        return 0.0f;

    const SparseVector& shorter = (a.count <= b.count) ? a : b;
    const SparseVector& longer = (a.count <= b.count) ? b : a;

    const auto lookup = index_map(shorter);

    float num = 0.0f, sum_sq_s = 0.0f, sum_sq_l = 0.0f;
    for (std::size_t i = 0; i < longer.count; ++i) {
        const auto idx = longer.indices[i];
        const auto it = lookup.find(idx);
        if (it == lookup.end())
            continue;
        const float m = mean_per_index[idx];
        const float ds = it->second - m;
        const float dl = longer.values[i] - m;
        num += ds * dl;
        sum_sq_s += ds * ds;
        sum_sq_l += dl * dl;
    }
    if (sum_sq_s <= 0.0f || sum_sq_l <= 0.0f)
        return 0.0f;
    return num / std::sqrt(sum_sq_s * sum_sq_l);
}

} // namespace recsys::similarity
