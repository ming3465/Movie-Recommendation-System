#include "recsys/similarity.hpp"

#include <cmath>
#include <unordered_map>

namespace recsys::similarity {

float cosine(const SparseVector& a, const SparseVector& b) {
    if (a.count == 0 || b.count == 0) return 0.0f;

    float norm_a_sq = 0.0f;
    for (std::size_t i = 0; i < a.count; ++i) {
        norm_a_sq += a.values[i] * a.values[i];
    }
    float norm_b_sq = 0.0f;
    for (std::size_t i = 0; i < b.count; ++i) {
        norm_b_sq += b.values[i] * b.values[i];
    }
    if (norm_a_sq <= 0.0f || norm_b_sq <= 0.0f) return 0.0f;

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

}  // namespace recsys::similarity
