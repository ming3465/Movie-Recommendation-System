#include <CLI/CLI.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "recsys/dataset.hpp"
#include "recsys/ratings_table.hpp"
#include "recsys/recommender.hpp"
#include "recsys/similarity.hpp"
#include "recsys/similarity_matrix.hpp"
#include "recsys/thread_pool.hpp"

int main(int argc, char** argv) {
    CLI::App app{"recsys - k-NN collaborative-filtering movie recommender"};
    app.set_version_flag("--version", "recsys 0.1.0");

    std::string dataset_path;
    int k = 30;
    std::string metric_str = "cosine";
    std::string mode_str = "ubcf";
    std::string output_path;
    float test_fraction = 0.2f;
    std::uint64_t seed = 42;
    std::size_t threads = 0;

    app.add_option("--dataset", dataset_path,
                   "Path to a MovieLens 100K u.data file")
        ->required();
    app.add_option("-k,--k", k, "Number of nearest neighbors")->default_val(30);
    app.add_option("-s,--similarity", metric_str,
                   "cosine | pearson | adjusted_cosine")
        ->default_val("cosine")
        ->check(CLI::IsMember({"cosine", "pearson", "adjusted_cosine"}));
    app.add_option("-m,--mode", mode_str,
                   "ubcf (user-based) | ibcf (item-based)")
        ->default_val("ubcf")
        ->check(CLI::IsMember({"ubcf", "ibcf"}));
    app.add_option("-o,--output", output_path,
                   "Write predictions CSV (user_id,item_id,actual,predicted)");
    app.add_option("--test-fraction", test_fraction,
                   "Held-out fraction of ratings")
        ->default_val(0.2f);
    app.add_option("--seed", seed, "RNG seed for the train/test split")
        ->default_val(42);
    app.add_option("--threads", threads,
                   "Worker threads (0 = hardware concurrency)")
        ->default_val(0);

    CLI11_PARSE(app, argc, argv);

    const recsys::Metric metric =
        (metric_str == "pearson") ? recsys::Metric::Pearson
        : (metric_str == "adjusted_cosine") ? recsys::Metric::AdjustedCosine
                                            : recsys::Metric::Cosine;
    const bool ubcf = (mode_str == "ubcf");

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    std::cout << "Loading " << dataset_path << "\n";
    const auto ratings = recsys::load_movielens_100k(dataset_path);
    std::cout << "  " << ratings.size() << " ratings\n";

    std::cout << "Splitting (test_fraction=" << test_fraction
              << ", seed=" << seed << ")\n";
    const auto split = recsys::train_test_split(ratings, test_fraction, seed);
    std::cout << "  train=" << split.train.size()
              << " test=" << split.test.size() << "\n";

    std::cout << "Building rating table\n";
    const recsys::RatingsTable rt(split.train);
    std::cout << "  users=" << rt.num_users()
              << " items=" << rt.num_items() << "\n";

    recsys::ThreadPool pool(threads);
    std::cout << "Building " << (ubcf ? "user-user" : "item-item")
              << " similarity matrix (" << metric_str << ") on "
              << pool.size() << " threads\n";
    const auto sim = ubcf ? recsys::build_user_similarity(rt, metric, pool)
                          : recsys::build_item_similarity(rt, metric, pool);

    std::cout << "Predicting " << split.test.size() << " test ratings (k="
              << k << ")\n";

    std::ofstream out;
    if (!output_path.empty()) {
        out.open(output_path);
        if (!out) {
            std::cerr << "Failed to open " << output_path << " for writing\n";
            return 1;
        }
        out << "user_id,item_id,actual,predicted\n";
    }

    std::size_t scored = 0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;

    auto score_one = [&](float p, const recsys::Rating& r) {
        const double err = static_cast<double>(p - r.rating);
        sum_abs += std::fabs(err);
        sum_sq += err * err;
        ++scored;
        if (out) {
            out << r.user_id << ',' << r.item_id << ',' << r.rating << ','
                << p << '\n';
        }
    };

    if (ubcf) {
        const recsys::UserKNN model(rt, sim, k);
        for (const auto& r : split.test) {
            const auto u = rt.user_index(r.user_id);
            const auto i = rt.item_index(r.item_id);
            if (u < 0 || i < 0) continue;  // unseen in training
            score_one(model.predict(u, i), r);
        }
    } else {
        const recsys::ItemKNN model(rt, sim, k);
        for (const auto& r : split.test) {
            const auto u = rt.user_index(r.user_id);
            const auto i = rt.item_index(r.item_id);
            if (u < 0 || i < 0) continue;
            score_one(model.predict(u, i), r);
        }
    }

    const auto t1 = clock::now();
    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Scored " << scored << "/" << split.test.size()
              << " test ratings in " << elapsed_s << "s\n";
    if (scored > 0) {
        const double mae = sum_abs / static_cast<double>(scored);
        const double rmse = std::sqrt(sum_sq / static_cast<double>(scored));
        std::cout << "  MAE  = " << mae << '\n';
        std::cout << "  RMSE = " << rmse << '\n';
    }
    if (!output_path.empty()) {
        std::cout << "Predictions written to " << output_path << '\n';
    }
    return 0;
}
