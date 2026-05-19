#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace recsys {

struct Rating {
    std::int32_t user_id;
    std::int32_t item_id;
    float rating;
};

struct Movie {
    std::int32_t id;
    std::string title;
};

// Parses a MovieLens 100K `u.data` file (TSV columns:
// user_id, item_id, rating, timestamp). The timestamp column is ignored.
// Throws std::runtime_error on missing file or malformed line.
std::vector<Rating> load_movielens_100k(const std::string& path);

// Parses a MovieLens 100K `u.item` file (pipe-separated; only the first two
// fields - movieId and title - are kept).
std::vector<Movie> load_movielens_100k_items(const std::string& path);

} // namespace recsys
