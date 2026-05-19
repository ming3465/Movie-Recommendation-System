#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <atomic>
#include <stdexcept>
#include <vector>

#include "recsys/thread_pool.hpp"

using recsys::ThreadPool;

TEST_CASE("ThreadPool: default size >= 1; explicit size honored") {
    ThreadPool def;
    CHECK(def.size() >= 1);
    ThreadPool four(4);
    CHECK(four.size() == 4);
}

TEST_CASE("ThreadPool: submit returns a future with the correct value") {
    ThreadPool pool(2);
    auto f = pool.submit([](int a, int b) { return a + b; }, 7, 35);
    CHECK(f.get() == 42);
}

TEST_CASE("ThreadPool: parallel_for visits every index exactly once") {
    ThreadPool pool(4);
    constexpr std::size_t N = 10'000;
    std::vector<int> visits(N, 0);
    pool.parallel_for(N, [&visits](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) visits[i] = 1;
    });
    long total = 0;
    for (auto v : visits) total += v;
    CHECK(total == static_cast<long>(N));
}

TEST_CASE("ThreadPool: parallel sum matches closed-form expected value") {
    ThreadPool pool(4);
    constexpr std::size_t N = 100'000;
    std::atomic<long long> sum{0};
    pool.parallel_for(N, [&sum](std::size_t b, std::size_t e) {
        long long local = 0;
        for (std::size_t i = b; i < e; ++i) local += static_cast<long long>(i);
        sum += local;
    });
    const long long expected = static_cast<long long>(N) * (N - 1) / 2;
    CHECK(sum.load() == expected);
}

TEST_CASE("ThreadPool: exception in a task propagates through future.get()") {
    ThreadPool pool(2);
    auto f = pool.submit([] { throw std::runtime_error("boom"); });
    CHECK_THROWS_AS(f.get(), std::runtime_error);
}

TEST_CASE("ThreadPool: exception in a parallel_for chunk is rethrown") {
    ThreadPool pool(4);
    CHECK_THROWS_AS(
        pool.parallel_for(
            100,
            [](std::size_t b, std::size_t /*e*/) {
                if (b == 0) throw std::runtime_error("chunk-failure");
            }),
        std::runtime_error);
}

TEST_CASE("ThreadPool: destructor drains in-flight tasks before joining") {
    std::atomic<int> done{0};
    {
        ThreadPool pool(2);
        for (int i = 0; i < 16; ++i) {
            pool.submit([&done] { ++done; });
        }
    }
    CHECK(done.load() == 16);
}
