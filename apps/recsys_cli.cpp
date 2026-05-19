#include <CLI/CLI.hpp>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"recsys - k-NN collaborative-filtering movie recommender"};
    app.set_version_flag("--version", "recsys 0.1.0");
    CLI11_PARSE(app, argc, argv);

    std::cout << "recsys 0.1.0 - no modules wired up yet. Try --help.\n";
    return 0;
}
