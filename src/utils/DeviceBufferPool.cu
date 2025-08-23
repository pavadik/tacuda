#include <utils/DeviceBufferPool.h>
#include <utils/CudaUtils.h>

DeviceBufferPool& DeviceBufferPool::instance() {
    static DeviceBufferPool pool;
    return pool;
}

void* DeviceBufferPool::acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    auto& vec = freeBuffers[bytes];
    if (!vec.empty()) {
        void* ptr = vec.back();
        vec.pop_back();
        return ptr;
    }
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    sizes[ptr] = bytes;
    return ptr;
}

void DeviceBufferPool::release(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = sizes.find(ptr);
    if (it != sizes.end()) {
        freeBuffers[it->second].push_back(ptr);
    }
}

void DeviceBufferPool::cleanup() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &kv : sizes) {
        cudaFree(kv.first);
    }
    freeBuffers.clear();
    sizes.clear();
}

DeviceBufferPool::~DeviceBufferPool() { cleanup(); }
