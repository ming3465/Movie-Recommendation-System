# Movie Recommendation System

[![ci](https://github.com/ming3465/Movie-Recommendation-System/actions/workflows/ci.yml/badge.svg)](https://github.com/ming3465/Movie-Recommendation-System/actions/workflows/ci.yml)

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
- Interactive search-and-recommend REPL (`-i`) - type a movie name, get neighbors
- doctest unit tests for every module; `ctest` integration via CMake
- `.clang-format` configured (LLVM, 100-col)

## Requirements

- CMake >= 3.14
- C++17 
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

### Interactive mode

Search for a movie by name (case-insensitive substring match) and get the
top 10 most similar films. Forces item-item similarity since "similar movies"
is naturally an item-side question.

```
./build/recsys_cli --dataset data/ml-100k/u.data --interactive
```

At the `Search>` prompt, type any movie name. The REPL prints matching
titles, then the top 10 similar films (with similarity scores) for the
first match. Quit with `/quit`, `/q`, or Ctrl+Z then Enter (Windows) /
Ctrl+D (Linux).

Example session:

```
Search> star wars
  Matched 1 title; showing 1:
    [1] Star Wars (1977)

  Top 10 movies similar to "Star Wars (1977)":
    1. Return of the Jedi (1983)         (sim=0.727)
    2. Raiders of the Lost Ark (1981)    (sim=0.634)
    3. Independence Day (ID4) (1996)     (sim=0.581)
    4. Empire Strikes Back, The (1980)   (sim=0.579)
    5. Toy Story (1995)                  (sim=0.577)
    6. Godfather, The (1972)             (sim=0.556)
    7. Indiana Jones and the Last Crusade (1989)  (sim=0.546)
    8. Fargo (1996)                      (sim=0.541)
    9. E.T. the Extra-Terrestrial (1982) (sim=0.532)
    10. Alien (1979)                     (sim=0.525)
```

Note: MovieLens 100K's catalog ends in early 1998, so "Avengers" and other
modern films aren't there. Try `star wars`, `toy story`, `pulp fiction`,
`godfather`, `titanic`, `fargo`, `jurassic park`.

Tests:

```
ctest --test-dir build --output-on-failure
```

## CLI options

| Flag                         | Default     | Description                            |
|------------------------------|-------------|----------------------------------------|
| `--dataset PATH`             | (required)  | MovieLens 100K `u.data` file           |
| `--items PATH`               | sibling of `--dataset` | `u.item` titles file (used by `-i`) |
| `-k, --k INT`                | 30          | Number of nearest neighbors            |
| `-s, --similarity NAME`      | cosine      | `cosine` / `pearson` / `adjusted_cosine` |
| `-m, --mode NAME`            | ubcf        | `ubcf` (user-based) / `ibcf` (item-based) |
| `-i, --interactive`          | off         | Drop into search-and-recommend REPL (forces IBCF) |
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
