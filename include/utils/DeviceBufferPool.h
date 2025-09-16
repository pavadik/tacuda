#ifndef DEVICE_BUFFER_POOL_H
#define DEVICE_BUFFER_POOL_H

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

class DeviceBufferPool {
public:
    static DeviceBufferPool& instance();

    void* acquire(size_t bytes);
    void release(void* ptr);

    // Frees all cached device buffers and resets the pool.
    void cleanup();

    // Sets the maximum amount of cached device memory in bytes.
    void setMaxCacheBytes(size_t bytes);

    // Number of cudaMalloc calls performed by the pool.
    size_t allocationCount() const;

    ~DeviceBufferPool();

private:
    DeviceBufferPool() = default;
    DeviceBufferPool(const DeviceBufferPool&) = delete;
    DeviceBufferPool& operator=(const DeviceBufferPool&) = delete;

    std::unordered_map<size_t, std::vector<void*>> freeBuffers;
    std::unordered_map<void*, size_t> sizes;
    size_t allocations = 0;
    size_t cachedBytes = 0;
    size_t maxCachedBytes = std::numeric_limits<size_t>::max();
    std::mutex mutex;
};

template <typename T>
struct DeviceBufferDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            DeviceBufferPool::instance().release(ptr);
        }
    }
};

template <typename T>
using DeviceBufferPtr = std::unique_ptr<T, DeviceBufferDeleter<T>>;

template <typename T>
DeviceBufferPtr<T> acquireDeviceBuffer(size_t count) {
    return DeviceBufferPtr<T>(static_cast<T*>(DeviceBufferPool::instance().acquire(count * sizeof(T))));
}

#endif
