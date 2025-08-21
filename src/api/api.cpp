#include <vector>
#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

#include "../../include/tacuda.h"
#include "../../include/indicators/SMA.h"
#include "../../include/indicators/Momentum.h"
#include "../../include/indicators/MACD.h"
#include "../../include/utils/CudaUtils.h"

extern "C" {

static int run_indicator(Indicator& ind, const float* h_in, float* h_out, int size) {
    try {
        float *d_in=nullptr, *d_out=nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice));

        ind.calculate(d_in, d_out, size);

        CUDA_CHECK(cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    } catch (...) {
        return 1;
    }
}

int ct_sma(const float* host_input, float* host_output, int size, int period) {
    SMA sma(period);
    return run_indicator(sma, host_input, host_output, size);
}

int ct_momentum(const float* host_input, float* host_output, int size, int period) {
    Momentum mom(period);
    return run_indicator(mom, host_input, host_output, size);
}

int ct_macd_line(const float* host_input, float* host_output, int size,
                 int fastPeriod, int slowPeriod, int signalPeriod) {
    MACD macd(fastPeriod, slowPeriod, signalPeriod);
    return run_indicator(macd, host_input, host_output, size);
}

} // extern "C"
