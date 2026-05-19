# Movie Recommendation System

A C++17 collaborative-filtering movie recommender. Builds a k-nearest-neighbors
model over a sparse ratings matrix and predicts unseen ratings with a
mean-centered weighted average of neighbor ratings. Benchmarked on
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/).

## Status

Early development. The CLI currently prints version info; the dataset loader,
sparse storage, similarity metrics, kNN recommender, and benchmark binary
will be added in subsequent commits.

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

```
./build/recsys_cli --version
```

(On a multi-config generator such as Visual Studio, the binary lives at
`build/Release/recsys_cli.exe`.)

## License

MIT - see [LICENSE](LICENSE).
