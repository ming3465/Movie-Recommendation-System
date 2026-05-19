// Benchmarks the recsys pipeline on MovieLens 100K across the six combinations
// of (UBCF, IBCF) x (cosine, pearson, adjusted_cosine). Prints a Markdown
// table to stdout; the README's "Benchmark" section is a hand-copy of one run.

#include <CLI/CLI.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "recsys/dataset.hpp"
#include "recsys/evaluator.hpp"
#include "recsys/ratings_table.hpp"
#include "recsys/recommender.hpp"
#include "recsys/similarity.hpp"
#include "recsys/similarity_matrix.hpp"
#include "recsys/thread_pool.hpp"

namespace {

struct Combo {
    const char* mode;
    const char* metric_label;
    bool ubcf;
    recsys::Metric metric;
};

struct Row {
    std::string mode;
    std::string metric;
    int k;
    double build_s;
    double predict_s;
    std::size_t scored;
    double rmse;
    double mae;
};

Row run_one(const Combo& c, int k, const recsys::RatingsTable& rt,
            const std::vector<recsys::Rating>& test, recsys::ThreadPool& pool) {
    using clock = std::chrono::steady_clock;

    const auto t0 = clock::now();
    const auto sim = c.ubcf ? recsys::build_user_similarity(rt, c.metric, pool)
                            : recsys::build_item_similarity(rt, c.metric, pool);
    const auto t1 = clock::now();

    recsys::Evaluator ev;
    if (c.ubcf) {
        const recsys::UserKNN model(rt, sim, k);
        for (const auto& r : test) {
            const auto u = rt.user_index(r.user_id);
            const auto i = rt.item_index(r.item_id);
            if (u < 0 || i < 0) continue;
            ev.add(model.predict(u, i), r.rating);
        }
    } else {
        const recsys::ItemKNN model(rt, sim, k);
        for (const auto& r : test) {
            const auto u = rt.user_index(r.user_id);
            const auto i = rt.item_index(r.item_id);
            if (u < 0 || i < 0) continue;
            ev.add(model.predict(u, i), r.rating);
        }
    }
    const auto t2 = clock::now();
    const auto m = ev.result();

    return Row{c.mode,
               c.metric_label,
               k,
               std::chrono::duration<double>(t1 - t0).count(),
               std::chrono::duration<double>(t2 - t1).count(),
               m.scored,
               m.rmse,
               m.mae};
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"bench_movielens100k - sweep (mode, metric) on MovieLens 100K"};
    std::string dataset_path = "data/ml-100k/u.data";
    int k = 30;
    std::uint64_t seed = 42;
    float test_fraction = 0.2f;
    std::size_t threads = 0;
    app.add_option("--dataset", dataset_path, "Path to MovieLens 100K u.data");
    app.add_option("-k,--k", k, "Neighbors")->default_val(30);
    app.add_option("--seed", seed, "Split seed")->default_val(42);
    app.add_option("--test-fraction", test_fraction, "Held-out fraction")->default_val(0.2f);
    app.add_option("--threads", threads, "Worker threads (0 = hw concurrency)")->default_val(0);
    CLI11_PARSE(app, argc, argv);

    std::cout << "# recsys bench - MovieLens 100K\n\n";
    std::cout << "dataset       = " << dataset_path << "\n";
    std::cout << "k             = " << k << "\n";
    std::cout << "test_fraction = " << test_fraction << "\n";
    std::cout << "seed          = " << seed << "\n";

    const auto ratings = recsys::load_movielens_100k(dataset_path);
    const auto split = recsys::train_test_split(ratings, test_fraction, seed);
    const recsys::RatingsTable rt(split.train);
    recsys::ThreadPool pool(threads);

    std::cout << "ratings       = " << ratings.size()
              << " (train=" << split.train.size() << ", test=" << split.test.size() << ")\n";
    std::cout << "train table   = " << rt.num_users() << " users x " << rt.num_items()
              << " items\n";
    std::cout << "threads       = " << pool.size() << "\n\n";

    const Combo combos[] = {
        {"UBCF", "cosine", true, recsys::Metric::Cosine},
        {"UBCF", "pearson", true, recsys::Metric::Pearson},
        {"UBCF", "adjusted_cosine", true, recsys::Metric::AdjustedCosine},
        {"IBCF", "cosine", false, recsys::Metric::Cosine},
        {"IBCF", "pearson", false, recsys::Metric::Pearson},
        {"IBCF", "adjusted_cosine", false, recsys::Metric::AdjustedCosine},
    };

    std::vector<Row> rows;
    rows.reserve(sizeof(combos) / sizeof(combos[0]));
    for (const auto& c : combos) {
        std::cerr << "  running " << c.mode << " / " << c.metric_label << "...\n";
        rows.push_back(run_one(c, k, rt, split.test, pool));
    }

    std::cout << "| Mode | Similarity      |  k | Build (s) | Predict (s) | Scored | RMSE  | MAE   "
                 "|\n";
    std::cout << "|------|-----------------|----|-----------|-------------|--------|-------|-------"
                 "|\n";
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& r : rows) {
        std::cout << "| " << r.mode << " | " << std::left << std::setw(15) << r.metric
                  << std::right << " | " << std::setw(2) << r.k << " | " << std::setw(9)
                  << r.build_s << " | " << std::setw(11) << r.predict_s << " | " << std::setw(6)
                  << r.scored << " | " << std::setw(5) << r.rmse << " | " << std::setw(5) << r.mae
                  << " |\n";
    }
    return 0;
}
