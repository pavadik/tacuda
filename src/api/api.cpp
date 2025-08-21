#include <vector>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>

#include "tacuda.h"
#include <indicators/SMA.h>
#include <indicators/Momentum.h>
#include <indicators/MACD.h>
#include <utils/CudaUtils.h>

extern "C" {

struct CudaDeleter {
    void operator()(float* ptr) const noexcept {
        if (ptr) cudaFree(ptr);
    }
};
using DeviceBuffer = std::unique_ptr<float, CudaDeleter>;

static ctStatus_t run_indicator(Indicator& ind, const float* h_in, float* h_out, int size) {
    DeviceBuffer d_in{nullptr}, d_out{nullptr};
    float* tmp = nullptr;

    cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_in.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_out.reset(tmp);

    err = cudaMemcpy(d_in.get(), h_in, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    try {
        ind.calculate(d_in.get(), d_out.get(), size);
    } catch (...) {
        return CT_STATUS_KERNEL_FAILED;
    }

    err = cudaMemcpy(h_out, d_out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    return CT_STATUS_SUCCESS;
}

ctStatus_t ct_sma(const float* host_input, float* host_output, int size, int period) {
    SMA sma(period);
    return run_indicator(sma, host_input, host_output, size);
}

ctStatus_t ct_momentum(const float* host_input, float* host_output, int size, int period) {
    Momentum mom(period);
    return run_indicator(mom, host_input, host_output, size);
}

ctStatus_t ct_macd_line(const float* host_input, float* host_output, int size,
                 int fastPeriod, int slowPeriod) {
    MACD macd(fastPeriod, slowPeriod);
    return run_indicator(macd, host_input, host_output, size);
}

} // extern "C"
