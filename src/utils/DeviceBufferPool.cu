#include <utils/DeviceBufferPool.h>
#include <utils/CudaUtils.h>

#include <exception>
#include <iostream>

DeviceBufferPool& DeviceBufferPool::instance() {
    static DeviceBufferPool pool;
    return pool;
}

void* DeviceBufferPool::acquire(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto& vec = freeBuffers[bytes];
    if (!vec.empty()) {
        void* ptr = vec.back();
        vec.pop_back();
        if (cachedBytes >= bytes) {
            cachedBytes -= bytes;
        } else {
            cachedBytes = 0;
        }
        return ptr;
    }
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    sizes[ptr] = bytes;
    ++allocations;
    return ptr;
}

void DeviceBufferPool::release(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = sizes.find(ptr);
    if (it != sizes.end()) {
        const size_t size = it->second;
        if (cachedBytes + size <= maxCachedBytes) {
            freeBuffers[size].push_back(ptr);
            cachedBytes += size;
        } else {
            CUDA_CHECK(cudaFree(ptr));
            sizes.erase(it);
        }
    }
}

void DeviceBufferPool::cleanup() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& kv : sizes) {
        CUDA_CHECK(cudaFree(kv.first));
    }
    freeBuffers.clear();
    sizes.clear();
    allocations = 0;
    cachedBytes = 0;
}

void DeviceBufferPool::setMaxCacheBytes(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    maxCachedBytes = bytes;
    if (cachedBytes <= maxCachedBytes) {
        return;
    }

    for (auto it = freeBuffers.begin(); it != freeBuffers.end() && cachedBytes > maxCachedBytes;) {
        auto& vec = it->second;
        const size_t bufferSize = it->first;
        while (!vec.empty() && cachedBytes > maxCachedBytes) {
            void* ptr = vec.back();
            vec.pop_back();
            CUDA_CHECK(cudaFree(ptr));
            cachedBytes -= bufferSize;
            sizes.erase(ptr);
        }
        if (vec.empty()) {
            it = freeBuffers.erase(it);
        } else {
            ++it;
        }
    }
}

DeviceBufferPool::~DeviceBufferPool() {
    try {
        cleanup();
    } catch (const std::exception& err) {
        std::cerr << "DeviceBufferPool cleanup failed during destruction: "
                  << err.what() << std::endl;
    }
}

size_t DeviceBufferPool::allocationCount() const {
    std::lock_guard<std::mutex> lock(mutex);
    return allocations;
}
