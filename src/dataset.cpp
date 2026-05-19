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

std::vector<Movie> load_movielens_100k_items(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open items file: " + path);
    }

    std::vector<Movie> movies;
    movies.reserve(2000);

    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty())
            continue;
        if (line.back() == '\r')
            line.pop_back();

        const auto p1 = line.find('|');
        if (p1 == std::string::npos) {
            throw std::runtime_error("Malformed item on line " + std::to_string(line_no) +
                                     " of " + path);
        }
        const auto p2 = line.find('|', p1 + 1);

        Movie m;
        try {
            m.id = std::stoi(line.substr(0, p1));
        } catch (const std::exception&) {
            throw std::runtime_error("Bad movie id on line " + std::to_string(line_no));
        }
        m.title = (p2 == std::string::npos) ? line.substr(p1 + 1)
                                            : line.substr(p1 + 1, p2 - p1 - 1);
        movies.push_back(std::move(m));
    }
    return movies;
}

} // namespace recsys
