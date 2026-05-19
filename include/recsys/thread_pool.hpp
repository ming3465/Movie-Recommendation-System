#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace recsys {

// Fixed-size pool of persistent worker threads. Submitted callables run on
// any worker; submit() returns a std::future for the result. All
// synchronization is RAII (lock_guard / unique_lock) - no public lock/unlock.
class ThreadPool {
public:
    // num_threads == 0 -> hardware_concurrency() (min 1).
    explicit ThreadPool(std::size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    std::size_t size() const noexcept { return workers_.size(); }

    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stop_) {
                throw std::runtime_error("submit on a stopped ThreadPool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    // Partition [0, n) across workers and call body(begin, end) on each
    // chunk in parallel. Blocks until every chunk finishes. If any chunk
    // throws, the first exception is rethrown after all chunks settle.
    void parallel_for(std::size_t n,
                      const std::function<void(std::size_t, std::size_t)>& body);

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

}  // namespace recsys
