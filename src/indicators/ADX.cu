#include <indicators/ADX.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void adxKernel(const float* __restrict__ high,
                          const float* __restrict__ low,
                          const float* __restrict__ close,
                          float* __restrict__ output,
                          int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2 * period - 1 || idx >= size) {
        return;
    }

    float prevHigh = high[0];
    float prevLow = low[0];
    float prevClose = close[0];
    float dmp_s = 0.0f;
    float dmm_s = 0.0f;
    float tr_s = 0.0f;
    float dx_sum = 0.0f;
    float adx = 0.0f;

    for (int i = 1; i <= idx; ++i) {
        float upMove = high[i] - prevHigh;
        float downMove = prevLow - low[i];
        float dmPlus = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
        float dmMinus = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
        float tr = fmaxf(high[i] - low[i],
                         fmaxf(fabsf(high[i] - prevClose),
                               fabsf(low[i] - prevClose)));

        prevHigh = high[i];
        prevLow = low[i];
        prevClose = close[i];

        if (i <= period) {
            dmp_s += dmPlus;
            dmm_s += dmMinus;
            tr_s += tr;
            if (i == period) {
                float dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
                float dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
                float dx = (dip + dim == 0.0f) ? 0.0f
                           : 100.0f * fabsf(dip - dim) / (dip + dim);
                dx_sum = dx;
            }
        } else {
            dmp_s = dmp_s - dmp_s / period + dmPlus;
            dmm_s = dmm_s - dmm_s / period + dmMinus;
            tr_s = tr_s - tr_s / period + tr;
            float dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
            float dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
            float dx = (dip + dim == 0.0f) ? 0.0f
                       : 100.0f * fabsf(dip - dim) / (dip + dim);
            if (i < 2 * period) {
                dx_sum += dx;
                if (i == 2 * period - 1) {
                    adx = dx_sum / period;
                }
            } else {
                adx = (adx * (period - 1) + dx) / period;
            }
        }
    }

    output[idx] = adx;
}

tacuda::ADX::ADX(int period) : period(period) {}

void tacuda::ADX::calculate(const float* high, const float* low, const float* close,
                    float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period > size) {
        throw std::invalid_argument("ADX: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    adxKernel<<<grid, block, 0, stream>>>(high, low, close, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::ADX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}
