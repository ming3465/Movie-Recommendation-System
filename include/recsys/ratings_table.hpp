#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "recsys/dataset.hpp"

namespace recsys {

// Compressed sparse storage of (user, item, rating) tuples with both
// user-major and item-major CSR views. External user/item ids are compacted
// to dense internal indices in [0, num_users()) and [0, num_items()).
class RatingsTable {
public:
    explicit RatingsTable(const std::vector<Rating>& ratings);

    std::int32_t num_users() const noexcept {
        return static_cast<std::int32_t>(user_off_.size() - 1);
    }
    std::int32_t num_items() const noexcept {
        return static_cast<std::int32_t>(item_off_.size() - 1);
    }
    std::size_t num_ratings() const noexcept { return user_items_.size(); }

    // External id -> internal index; -1 if the id was never seen.
    std::int32_t user_index(std::int32_t ext_user_id) const;
    std::int32_t item_index(std::int32_t ext_item_id) const;

    // Non-owning view of one CSR row.
    struct UserSlice {
        const std::int32_t* items;
        const float* values;
        std::size_t count;
    };
    struct ItemSlice {
        const std::int32_t* users;
        const float* values;
        std::size_t count;
    };

    UserSlice ratings_by_user(std::int32_t user_idx) const;
    ItemSlice ratings_by_item(std::int32_t item_idx) const;

    float user_mean(std::int32_t user_idx) const noexcept {
        return user_means_[static_cast<std::size_t>(user_idx)];
    }
    float item_mean(std::int32_t item_idx) const noexcept {
        return item_means_[static_cast<std::size_t>(item_idx)];
    }
    const float* user_means_data() const noexcept { return user_means_.data(); }
    const float* item_means_data() const noexcept { return item_means_.data(); }
    float global_mean() const noexcept { return global_mean_; }

private:
    std::vector<std::size_t> user_off_;
    std::vector<std::int32_t> user_items_;
    std::vector<float> user_values_;

    std::vector<std::size_t> item_off_;
    std::vector<std::int32_t> item_users_;
    std::vector<float> item_values_;

    std::vector<float> user_means_;
    std::vector<float> item_means_;
    float global_mean_ = 0.0f;

    std::unordered_map<std::int32_t, std::int32_t> user_ext_to_int_;
    std::unordered_map<std::int32_t, std::int32_t> item_ext_to_int_;
};

// Random partition into (train, test). Each input rating is placed in `test`
// with probability `test_fraction`; the same `seed` gives the same split.
struct Split {
    std::vector<Rating> train;
    std::vector<Rating> test;
};

Split train_test_split(const std::vector<Rating>& ratings,
                       float test_fraction,
                       std::uint64_t seed);

}  // namespace recsys
