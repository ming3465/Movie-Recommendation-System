#include "recsys/thread_pool.hpp"

#include <algorithm>
#include <exception>

namespace recsys {

ThreadPool::ThreadPool(std::size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    workers_.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable())
            w.join();
    }
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty())
                return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for(std::size_t n,
                              const std::function<void(std::size_t, std::size_t)>& body) {
    if (n == 0)
        return;
    const std::size_t threads = std::max<std::size_t>(1, workers_.size());
    const std::size_t chunk = (n + threads - 1) / threads;

    std::vector<std::future<void>> futs;
    futs.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        const std::size_t b = t * chunk;
        if (b >= n)
            break;
        const std::size_t e = std::min(n, b + chunk);
        futs.emplace_back(submit([&body, b, e] { body(b, e); }));
    }

    std::exception_ptr first_exc;
    for (auto& f : futs) {
        try {
            f.get();
        } catch (...) {
            if (!first_exc)
                first_exc = std::current_exception();
        }
    }
    if (first_exc)
        std::rethrow_exception(first_exc);
}

} // namespace recsys
