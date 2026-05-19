#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "recsys/dataset.hpp"

int main(int argc, char** argv) {
    CLI::App app{"recsys - k-NN collaborative-filtering movie recommender"};
    app.set_version_flag("--version", "recsys 0.1.0");

    std::string dataset_path;
    app.add_option("--dataset", dataset_path,
                   "Path to a MovieLens 100K u.data file. Loads and reports count.");

    CLI11_PARSE(app, argc, argv);

    if (!dataset_path.empty()) {
        const auto ratings = recsys::load_movielens_100k(dataset_path);
        std::cout << "Loaded " << ratings.size() << " ratings from " << dataset_path << "\n";
        return 0;
    }

    std::cout << "recsys 0.1.0 - pass --dataset <path> to load ratings, or --help.\n";
    return 0;
}
