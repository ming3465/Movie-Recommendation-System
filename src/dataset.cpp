#include "recsys/dataset.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace recsys {

std::vector<Rating> load_movielens_100k(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open dataset: " + path);
    }

    std::vector<Rating> ratings;
    ratings.reserve(100000);

    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty())
            continue;

        std::istringstream is(line);
        Rating r{};
        if (!(is >> r.user_id >> r.item_id >> r.rating)) {
            throw std::runtime_error("Malformed rating on line " + std::to_string(line_no) +
                                     " of " + path);
        }
        ratings.push_back(r);
    }
    return ratings;
}

} // namespace recsys
