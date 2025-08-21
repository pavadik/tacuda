#include <vector>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>

#include "tacuda.h"
#include <indicators/SMA.h>
#include <indicators/Momentum.h>
#include <indicators/MACD.h>
#include <indicators/EMA.h>
#include <indicators/WMA.h>
#include <indicators/RSI.h>
#include <indicators/BBANDS.h>
#include <indicators/ATR.h>
#include <indicators/Stochastic.h>
#include <indicators/CCI.h>
#include <indicators/OBV.h>
#include <utils/CudaUtils.h>

extern "C" {

struct CudaDeleter {
    void operator()(float* ptr) const noexcept {
        if (ptr) cudaFree(ptr);
    }
};
using DeviceBuffer = std::unique_ptr<float, CudaDeleter>;

static ctStatus_t run_indicator(Indicator& ind, const float* h_in, float* h_out,
                                int size, int outMultiple = 1) {
    DeviceBuffer d_in{nullptr}, d_out{nullptr};
    float* tmp = nullptr;

    cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_in.reset(tmp);

    err = cudaMalloc(&tmp, size * outMultiple * sizeof(float));
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

    err = cudaMemcpy(h_out, d_out.get(), size * outMultiple * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    return CT_STATUS_SUCCESS;
}

ctStatus_t ct_sma(const float* host_input, float* host_output, int size, int period) {
    SMA sma(period);
    return run_indicator(sma, host_input, host_output, size);
}

ctStatus_t ct_wma(const float* host_input, float* host_output, int size, int period) {
    WMA wma(period);
    return run_indicator(wma, host_input, host_output, size);
}

ctStatus_t ct_momentum(const float* host_input, float* host_output, int size, int period) {
    Momentum mom(period);
    return run_indicator(mom, host_input, host_output, size);
}

ctStatus_t ct_ema(const float* host_input, float* host_output, int size, int period) {
    EMA ema(period);
    return run_indicator(ema, host_input, host_output, size);
}

ctStatus_t ct_rsi(const float* host_input, float* host_output, int size, int period) {
    RSI rsi(period);
    return run_indicator(rsi, host_input, host_output, size);
}

ctStatus_t ct_macd_line(const float* host_input, float* host_output, int size,
                 int fastPeriod, int slowPeriod) {
    MACD macd(fastPeriod, slowPeriod);
    return run_indicator(macd, host_input, host_output, size);
}

ctStatus_t ct_bbands(const float* host_input,
                     float* host_upper,
                     float* host_middle,
                     float* host_lower,
                     int size,
                     int period,
                     float upperMul,
                     float lowerMul) {
    BBANDS bb(period, upperMul, lowerMul);
    std::vector<float> tmp(3 * size);
    ctStatus_t rc = run_indicator(bb, host_input, tmp.data(), size, 3);
    if (rc != CT_STATUS_SUCCESS) {
        return rc;
    }
    std::memcpy(host_upper, tmp.data(), size * sizeof(float));
    std::memcpy(host_middle, tmp.data() + size, size * sizeof(float));
    std::memcpy(host_lower, tmp.data() + 2 * size, size * sizeof(float));
    return CT_STATUS_SUCCESS;
}

ctStatus_t ct_atr(const float* host_high,
                  const float* host_low,
                  const float* host_close,
                  float* host_output,
                  int size,
                  int period,
                  float initial) {
    ATR atr(period, initial);
    DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr}, d_out{nullptr};
    float* tmp = nullptr;

    cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_high.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_low.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_close.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_out.reset(tmp);

    err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    try {
        atr.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
    } catch (...) {
        return CT_STATUS_KERNEL_FAILED;
    }

    err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    return CT_STATUS_SUCCESS;
}

ctStatus_t ct_stochastic(const float* host_high,
                         const float* host_low,
                         const float* host_close,
                         float* host_k,
                         float* host_d,
                         int size,
                         int kPeriod,
                         int dPeriod) {
    Stochastic stoch(kPeriod, dPeriod);
    DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr}, d_out{nullptr};
    float* tmp = nullptr;

    cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_high.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_low.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_close.reset(tmp);

    err = cudaMalloc(&tmp, 2 * size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_out.reset(tmp);

    err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    try {
        stoch.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
    } catch (...) {
        return CT_STATUS_KERNEL_FAILED;
    }

    std::vector<float> tmpHost(2 * size);
    err = cudaMemcpy(tmpHost.data(), d_out.get(), 2 * size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    std::memcpy(host_k, tmpHost.data(), size * sizeof(float));
    std::memcpy(host_d, tmpHost.data() + size, size * sizeof(float));

    return CT_STATUS_SUCCESS;
}

ctStatus_t ct_cci(const float* host_high,
                  const float* host_low,
                  const float* host_close,
                  float* host_output,
                  int size,
                  int period) {
    CCI cci(period);
    DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr}, d_out{nullptr};
    float* tmp = nullptr;

    cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_high.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_low.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_close.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_out.reset(tmp);

    err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    try {
        cci.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
    } catch (...) {
        return CT_STATUS_KERNEL_FAILED;
    }

    err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    return CT_STATUS_SUCCESS;
}

ctStatus_t ct_obv(const float* host_price,
                  const float* host_volume,
                  float* host_output,
                  int size) {
    OBV obv;
    DeviceBuffer d_price{nullptr}, d_volume{nullptr}, d_out{nullptr};
    float* tmp = nullptr;

    cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_price.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_volume.reset(tmp);

    err = cudaMalloc(&tmp, size * sizeof(float));
    if (err != cudaSuccess) {
        return CT_STATUS_ALLOC_FAILED;
    }
    d_out.reset(tmp);

    err = cudaMemcpy(d_price.get(), host_price, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaMemcpy(d_volume.get(), host_volume, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    try {
        obv.calculate(d_price.get(), d_volume.get(), d_out.get(), size);
    } catch (...) {
        return CT_STATUS_KERNEL_FAILED;
    }

    err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return CT_STATUS_COPY_FAILED;
    }

    return CT_STATUS_SUCCESS;
}

} // extern "C"
