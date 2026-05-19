#include "recsys/ratings_table.hpp"

#include <random>
#include <stdexcept>

namespace recsys {

namespace {

std::unordered_map<std::int32_t, std::int32_t> compact_ids(const std::vector<Rating>& ratings,
                                                           bool is_user) {
    std::unordered_map<std::int32_t, std::int32_t> map;
    map.reserve(ratings.size());
    for (const auto& r : ratings) {
        const std::int32_t ext = is_user ? r.user_id : r.item_id;
        if (map.find(ext) == map.end()) {
            map.emplace(ext, static_cast<std::int32_t>(map.size()));
        }
    }
    return map;
}

} // namespace

RatingsTable::RatingsTable(const std::vector<Rating>& ratings) {
    user_ext_to_int_ = compact_ids(ratings, /*is_user=*/true);
    item_ext_to_int_ = compact_ids(ratings, /*is_user=*/false);

    const auto n_users = static_cast<std::size_t>(user_ext_to_int_.size());
    const auto n_items = static_cast<std::size_t>(item_ext_to_int_.size());
    const auto n = ratings.size();

    // ---- User-major CSR ----
    user_off_.assign(n_users + 1, 0);
    for (const auto& r : ratings) {
        ++user_off_[static_cast<std::size_t>(user_ext_to_int_[r.user_id]) + 1];
    }
    for (std::size_t u = 1; u <= n_users; ++u) {
        user_off_[u] += user_off_[u - 1];
    }
    user_items_.resize(n);
    user_values_.resize(n);
    {
        std::vector<std::size_t> cursor = user_off_;
        for (const auto& r : ratings) {
            const auto u = static_cast<std::size_t>(user_ext_to_int_[r.user_id]);
            const auto i = item_ext_to_int_[r.item_id];
            const auto pos = cursor[u]++;
            user_items_[pos] = i;
            user_values_[pos] = r.rating;
        }
    }

    // ---- Item-major CSR ----
    item_off_.assign(n_items + 1, 0);
    for (const auto& r : ratings) {
        ++item_off_[static_cast<std::size_t>(item_ext_to_int_[r.item_id]) + 1];
    }
    for (std::size_t i = 1; i <= n_items; ++i) {
        item_off_[i] += item_off_[i - 1];
    }
    item_users_.resize(n);
    item_values_.resize(n);
    {
        std::vector<std::size_t> cursor = item_off_;
        for (const auto& r : ratings) {
            const auto u = user_ext_to_int_[r.user_id];
            const auto i = static_cast<std::size_t>(item_ext_to_int_[r.item_id]);
            const auto pos = cursor[i]++;
            item_users_[pos] = u;
            item_values_[pos] = r.rating;
        }
    }

    // ---- Means ----
    user_means_.assign(n_users, 0.0f);
    for (std::size_t u = 0; u < n_users; ++u) {
        const auto first = user_off_[u];
        const auto last = user_off_[u + 1];
        if (first == last)
            continue;
        float sum = 0.0f;
        for (auto p = first; p < last; ++p)
            sum += user_values_[p];
        user_means_[u] = sum / static_cast<float>(last - first);
    }
    item_means_.assign(n_items, 0.0f);
    for (std::size_t i = 0; i < n_items; ++i) {
        const auto first = item_off_[i];
        const auto last = item_off_[i + 1];
        if (first == last)
            continue;
        float sum = 0.0f;
        for (auto p = first; p < last; ++p)
            sum += item_values_[p];
        item_means_[i] = sum / static_cast<float>(last - first);
    }

    if (!ratings.empty()) {
        double sum = 0.0;
        for (const auto& r : ratings)
            sum += r.rating;
        global_mean_ = static_cast<float>(sum / static_cast<double>(n));
    }
}

std::int32_t RatingsTable::user_index(std::int32_t ext_user_id) const {
    const auto it = user_ext_to_int_.find(ext_user_id);
    return it == user_ext_to_int_.end() ? -1 : it->second;
}

std::int32_t RatingsTable::item_index(std::int32_t ext_item_id) const {
    const auto it = item_ext_to_int_.find(ext_item_id);
    return it == item_ext_to_int_.end() ? -1 : it->second;
}

RatingsTable::UserSlice RatingsTable::ratings_by_user(std::int32_t user_idx) const {
    if (user_idx < 0 || user_idx >= num_users()) {
        throw std::out_of_range("ratings_by_user: user index out of range");
    }
    const auto u = static_cast<std::size_t>(user_idx);
    const auto first = user_off_[u];
    const auto last = user_off_[u + 1];
    return {user_items_.data() + first, user_values_.data() + first, last - first};
}

RatingsTable::ItemSlice RatingsTable::ratings_by_item(std::int32_t item_idx) const {
    if (item_idx < 0 || item_idx >= num_items()) {
        throw std::out_of_range("ratings_by_item: item index out of range");
    }
    const auto i = static_cast<std::size_t>(item_idx);
    const auto first = item_off_[i];
    const auto last = item_off_[i + 1];
    return {item_users_.data() + first, item_values_.data() + first, last - first};
}

Split train_test_split(const std::vector<Rating>& ratings, float test_fraction,
                       std::uint64_t seed) {
    if (test_fraction < 0.0f || test_fraction > 1.0f) {
        throw std::invalid_argument("test_fraction must be in [0, 1]");
    }
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    Split out;
    out.train.reserve(ratings.size());
    out.test.reserve(ratings.size() / 4);

    for (const auto& r : ratings) {
        if (uni(rng) < test_fraction) {
            out.test.push_back(r);
        } else {
            out.train.push_back(r);
        }
    }
    return out;
}

} // namespace recsys
