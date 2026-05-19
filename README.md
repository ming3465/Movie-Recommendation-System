# Movie Recommendation System

A C++17 collaborative-filtering movie recommender. Builds a k-nearest-neighbors
model over a sparse ratings matrix and predicts unseen ratings with a
mean-centered weighted average of neighbor ratings. Benchmarked on
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/).

## Highlights

- User-based and item-based kNN, both with mean-centered prediction
- Cosine, Pearson, and adjusted-cosine similarity metrics
- Sparse CSR storage with user-major and item-major views
- Custom thread pool with RAII locking and futures; parallel similarity matrix
- Seeded train/test split for reproducible evaluation
- doctest unit tests for every module; `ctest` integration via CMake
- `.clang-format` configured (LLVM, 100-col)

## Requirements

- CMake >= 3.14
- A C++17 compiler (MSVC 2019+, GCC 8+, or Clang 7+)
- Network access on first configure (CMake fetches `doctest` and `CLI11`)

## Build

```
cmake -B build -S .
cmake --build build --config Release
```

## Run

Download the dataset (one-time):

```
cmake --build build --target download_movielens_100k
```

Run a single configuration:

```
./build/recsys_cli --dataset data/ml-100k/u.data -k 30 -s cosine -m ubcf
```

Sweep all (mode, metric) combinations and print a Markdown results table:

```
./build/bench_movielens100k
```

Tests:

```
ctest --test-dir build --output-on-failure
```

## CLI options

| Flag                         | Default     | Description                            |
|------------------------------|-------------|----------------------------------------|
| `--dataset PATH`             | (required)  | MovieLens 100K `u.data` file           |
| `-k, --k INT`                | 30          | Number of nearest neighbors            |
| `-s, --similarity NAME`      | cosine      | `cosine` / `pearson` / `adjusted_cosine` |
| `-m, --mode NAME`            | ubcf        | `ubcf` (user-based) / `ibcf` (item-based) |
| `-o, --output PATH`          | (none)      | Write predictions CSV here             |
| `--test-fraction FLOAT`      | 0.2         | Held-out fraction of ratings           |
| `--seed UINT`                | 42          | RNG seed for the train/test split      |
| `--threads UINT`             | 0 (auto)    | Worker threads                         |

## Benchmark

MovieLens 100K, k=30, 80/20 split (seed=42), 12 worker threads. One run from
`./build/bench_movielens100k`:

| Mode | Similarity      |  k | Build (s) | Predict (s) | Scored | RMSE  | MAE   |
|------|-----------------|----|-----------|-------------|--------|-------|-------|
| UBCF | cosine          | 30 |     1.886 |       0.201 |  20034 | 0.931 | 0.730 |
| UBCF | pearson         | 30 |     1.672 |       0.206 |  20034 | 0.934 | 0.735 |
| UBCF | adjusted_cosine | 30 |     1.785 |       0.199 |  20034 | 0.973 | 0.764 |
| IBCF | cosine          | 30 |     3.498 |       0.254 |  20034 | 0.911 | 0.716 |
| IBCF | pearson         | 30 |     3.431 |       0.250 |  20034 | 0.938 | 0.738 |
| IBCF | adjusted_cosine | 30 |     3.347 |       0.248 |  20034 | 0.934 | 0.735 |

Best configuration on this split: **IBCF + cosine, RMSE 0.911**. The numbers
sit in the band published for cosine kNN on MovieLens 100K (~0.93-0.99
depending on the split and mean-centering scheme).

## Project layout

```
.
├── apps/recsys_cli.cpp                 CLI entry point
├── benchmarks/bench_movielens100k.cpp  Sweep across (mode, metric)
├── cmake/DownloadMovieLens.cmake       Dataset download script
├── include/recsys/                     Public headers
│   ├── dataset.hpp
│   ├── ratings_table.hpp
│   ├── thread_pool.hpp
│   ├── similarity.hpp
│   ├── similarity_matrix.hpp
│   ├── recommender.hpp
│   └── evaluator.hpp
├── src/                                Implementations
└── tests/                              doctest unit tests
```

## License

MIT - see [LICENSE](LICENSE).
