#include <indicators/DX.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <math.h>

__global__ void dxKernel(const float* __restrict__ high,
                         const float* __restrict__ low,
                         const float* __restrict__ close,
                         float* __restrict__ output,
                         int period, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - period) {
        float prevHigh = high[idx];
        float prevLow = low[idx];
        float prevClose = close[idx];
        float dmp = 0.0f, dmm = 0.0f, tr = 0.0f;
        for (int i = 1; i <= period; ++i) {
            float curHigh = high[idx + i];
            float curLow = low[idx + i];
            float upMove = curHigh - prevHigh;
            float downMove = prevLow - curLow;
            float dmPlus = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
            float dmMinus = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
            float trVal = fmaxf(curHigh - curLow,
                                fmaxf(fabsf(curHigh - prevClose),
                                      fabsf(curLow - prevClose)));
            dmp += dmPlus;
            dmm += dmMinus;
            tr += trVal;
            prevHigh = curHigh;
            prevLow = curLow;
            prevClose = close[idx + i];
        }
        float dip = (tr == 0.0f) ? 0.0f : 100.0f * dmp / tr;
        float dim = (tr == 0.0f) ? 0.0f : 100.0f * dmm / tr;
        float denom = dip + dim;
        output[idx] = (denom == 0.0f) ? 0.0f : 100.0f * fabsf(dip - dim) / denom;
    }
}

DX::DX(int period) : period(period) {}

void DX::calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0 || period >= size) {
        throw std::invalid_argument("DX: invalid period");
    }
    CUDA_CHECK(cudaMemset(output, 0xFF, size * sizeof(float)));
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    dxKernel<<<grid, block, 0, stream>>>(high, low, close, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void DX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high = input;
    const float* low = input + size;
    const float* close = input + 2 * size;
    calculate(high, low, close, output, size, stream);
}

