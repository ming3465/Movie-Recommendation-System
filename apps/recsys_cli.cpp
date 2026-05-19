#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "recsys/dataset.hpp"
#include "recsys/ratings_table.hpp"
#include "recsys/recommender.hpp"
#include "recsys/similarity.hpp"
#include "recsys/similarity_matrix.hpp"
#include "recsys/thread_pool.hpp"

namespace {

std::string to_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string default_items_path(const std::string& dataset_path) {
    // Heuristic: replace trailing "u.data" with "u.item".
    const std::string needle = "u.data";
    const auto pos = dataset_path.rfind(needle);
    if (pos == std::string::npos || pos + needle.size() != dataset_path.size()) {
        return "data/ml-100k/u.item";
    }
    return dataset_path.substr(0, pos) + "u.item";
}

void run_interactive(const recsys::RatingsTable& rt, const recsys::SimilarityMatrix& sim,
                     const std::vector<recsys::Movie>& movies) {
    // Map from internal item index -> title (only items present in training).
    std::vector<std::string> title_by_idx(static_cast<std::size_t>(rt.num_items()));
    for (const auto& m : movies) {
        const auto ii = rt.item_index(m.id);
        if (ii >= 0 && static_cast<std::size_t>(ii) < title_by_idx.size()) {
            title_by_idx[static_cast<std::size_t>(ii)] = m.title;
        }
    }

    std::cout << "\nInteractive mode. Type a movie name (case-insensitive).\n";
    std::cout << "Type /quit or Ctrl+D / Ctrl+Z to exit.\n";

    while (true) {
        std::cout << "\nSearch> " << std::flush;
        std::string query;
        if (!std::getline(std::cin, query)) {
            std::cout << "\n";
            break;
        }
        while (!query.empty() && (query.back() == '\r' || query.back() == ' ')) {
            query.pop_back();
        }
        if (query.empty()) continue;
        if (query == "/quit" || query == "/q" || query == "/exit") break;

        const std::string q_lower = to_lower(query);

        // Case-insensitive substring search across all titles.
        std::vector<const recsys::Movie*> matches;
        for (const auto& m : movies) {
            if (to_lower(m.title).find(q_lower) != std::string::npos) {
                matches.push_back(&m);
            }
        }

        if (matches.empty()) {
            std::cout << "  No movies matched \"" << query << "\".\n";
            continue;
        }

        const std::size_t show = std::min<std::size_t>(matches.size(), 8);
        std::cout << "  Matched " << matches.size() << " title"
                  << (matches.size() == 1 ? "" : "s") << "; showing " << show << ":\n";
        for (std::size_t i = 0; i < show; ++i) {
            std::cout << "    [" << (i + 1) << "] " << matches[i]->title << "\n";
        }

        // Auto-pick the first match for the recommendation step.
        const recsys::Movie* picked = matches[0];
        const auto pidx = rt.item_index(picked->id);
        if (pidx < 0) {
            std::cout << "  \"" << picked->title
                      << "\" has no ratings in the training split; no neighbors.\n";
            continue;
        }

        // Top-N similar items by similarity score.
        struct Cand {
            float sim;
            std::int32_t idx;
        };
        std::vector<Cand> cands;
        cands.reserve(static_cast<std::size_t>(rt.num_items()));
        for (std::int32_t j = 0; j < rt.num_items(); ++j) {
            if (j == pidx) continue;
            const float s = sim.at(pidx, j);
            if (s > 0.0f) cands.push_back({s, j});
        }

        constexpr std::size_t kTop = 10;
        const std::size_t k = std::min(kTop, cands.size());
        std::partial_sort(cands.begin(),
                          cands.begin() + static_cast<std::ptrdiff_t>(k), cands.end(),
                          [](const Cand& a, const Cand& b) { return a.sim > b.sim; });

        std::cout << "\n  Top " << k << " movies similar to \"" << picked->title << "\":\n";
        for (std::size_t i = 0; i < k; ++i) {
            const auto& c = cands[i];
            const auto& title = title_by_idx[static_cast<std::size_t>(c.idx)];
            std::cout << "    " << (i + 1) << ". "
                      << (title.empty() ? "<unknown>" : title) << "    (sim=" << c.sim << ")\n";
        }
    }
    std::cout << "Bye.\n";
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"recsys - k-NN collaborative-filtering movie recommender"};
    app.set_version_flag("--version", "recsys 0.1.0");

    std::string dataset_path;
    std::string items_path;
    int k = 30;
    std::string metric_str = "cosine";
    std::string mode_str = "ubcf";
    std::string output_path;
    float test_fraction = 0.2f;
    std::uint64_t seed = 42;
    std::size_t threads = 0;
    bool interactive = false;

    app.add_option("--dataset", dataset_path, "Path to a MovieLens 100K u.data file")
        ->required();
    app.add_option("--items", items_path,
                   "Path to u.item (titles). Defaults to sibling of --dataset.");
    app.add_option("-k,--k", k, "Number of nearest neighbors")->default_val(30);
    app.add_option("-s,--similarity", metric_str, "cosine | pearson | adjusted_cosine")
        ->default_val("cosine")
        ->check(CLI::IsMember({"cosine", "pearson", "adjusted_cosine"}));
    app.add_option("-m,--mode", mode_str, "ubcf (user-based) | ibcf (item-based)")
        ->default_val("ubcf")
        ->check(CLI::IsMember({"ubcf", "ibcf"}));
    app.add_option("-o,--output", output_path,
                   "Write predictions CSV (user_id,item_id,actual,predicted)");
    app.add_option("--test-fraction", test_fraction, "Held-out fraction of ratings")
        ->default_val(0.2f);
    app.add_option("--seed", seed, "RNG seed for the train/test split")->default_val(42);
    app.add_option("--threads", threads, "Worker threads (0 = hardware concurrency)")
        ->default_val(0);
    app.add_flag("-i,--interactive", interactive,
                 "Interactive search-and-recommend REPL (forces IBCF + cosine)");

    CLI11_PARSE(app, argc, argv);

    // Interactive mode overrides a few options for a coherent UX.
    if (interactive) {
        mode_str = "ibcf";
    }

    const recsys::Metric metric = (metric_str == "pearson") ? recsys::Metric::Pearson
                                  : (metric_str == "adjusted_cosine")
                                      ? recsys::Metric::AdjustedCosine
                                      : recsys::Metric::Cosine;
    const bool ubcf = (mode_str == "ubcf");

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    std::cout << "Loading " << dataset_path << "\n";
    const auto ratings = recsys::load_movielens_100k(dataset_path);
    std::cout << "  " << ratings.size() << " ratings\n";

    std::vector<recsys::Movie> movies;
    if (interactive) {
        const std::string ip = items_path.empty() ? default_items_path(dataset_path) : items_path;
        std::cout << "Loading titles from " << ip << "\n";
        movies = recsys::load_movielens_100k_items(ip);
        std::cout << "  " << movies.size() << " titles\n";
    }

    std::cout << "Splitting (test_fraction=" << test_fraction << ", seed=" << seed << ")\n";
    const auto split = recsys::train_test_split(ratings, test_fraction, seed);
    std::cout << "  train=" << split.train.size() << " test=" << split.test.size() << "\n";

    std::cout << "Building rating table\n";
    const recsys::RatingsTable rt(split.train);
    std::cout << "  users=" << rt.num_users() << " items=" << rt.num_items() << "\n";

    recsys::ThreadPool pool(threads);
    std::cout << "Building " << (ubcf ? "user-user" : "item-item") << " similarity matrix ("
              << metric_str << ") on " << pool.size() << " threads\n";
    const auto sim = ubcf ? recsys::build_user_similarity(rt, metric, pool)
                          : recsys::build_item_similarity(rt, metric, pool);

    const auto t1 = clock::now();
    std::cout << "Ready in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    if (interactive) {
        run_interactive(rt, sim, movies);
        return 0;
    }

    std::cout << "Predicting " << split.test.size() << " test ratings (k=" << k << ")\n";

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
            out << r.user_id << ',' << r.item_id << ',' << r.rating << ',' << p << '\n';
        }
    };

    if (ubcf) {
        const recsys::UserKNN model(rt, sim, k);
        for (const auto& r : split.test) {
            const auto u = rt.user_index(r.user_id);
            const auto i = rt.item_index(r.item_id);
            if (u < 0 || i < 0) continue;
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

    const auto t2 = clock::now();
    const double elapsed_s = std::chrono::duration<double>(t2 - t0).count();

    std::cout << "Scored " << scored << "/" << split.test.size() << " test ratings in "
              << elapsed_s << "s\n";
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
