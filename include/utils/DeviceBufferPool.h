#ifndef DEVICE_BUFFER_POOL_H
#define DEVICE_BUFFER_POOL_H

#include <cstddef>
#include <unordered_map>
#include <vector>
#include <mutex>

class DeviceBufferPool {
public:
    static DeviceBufferPool& instance();

    void* acquire(size_t bytes);
    void release(void* ptr);

    ~DeviceBufferPool();

private:
    DeviceBufferPool() = default;
    DeviceBufferPool(const DeviceBufferPool&) = delete;
    DeviceBufferPool& operator=(const DeviceBufferPool&) = delete;

    std::unordered_map<size_t, std::vector<void*>> freeBuffers;
    std::unordered_map<void*, size_t> sizes;
    std::mutex mutex;
};

#endif
