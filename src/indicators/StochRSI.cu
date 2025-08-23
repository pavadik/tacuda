#include <indicators/StochRSI.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

__global__ void stochRsiKernel(const float *__restrict__ input,
                               float *__restrict__ rsi,
                               float *__restrict__ kOut,
                               float *__restrict__ dOut, int rsiPeriod,
                               int kPeriod, int dPeriod, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float nan = nanf("");
    for (int i = 0; i < size; ++i) {
      rsi[i] = nan;
      kOut[i] = nan;
      dOut[i] = nan;
    }
    for (int i = rsiPeriod; i < size; ++i) {
      float gain = 0.0f, loss = 0.0f;
      for (int j = 0; j < rsiPeriod; ++j) {
        float diff = input[i - j] - input[i - j - 1];
        if (diff > 0.0f)
          gain += diff;
        else
          loss -= diff;
      }
      float avgGain = gain / rsiPeriod;
      float avgLoss = loss / rsiPeriod;
      float val;
      if (avgLoss == 0.0f)
        val = (avgGain == 0.0f) ? 50.0f : 100.0f;
      else if (avgGain == 0.0f)
        val = 0.0f;
      else {
        float rs = avgGain / avgLoss;
        val = 100.0f - 100.0f / (1.0f + rs);
      }
      rsi[i] = val;
      if (i >= rsiPeriod + kPeriod - 1) {
        float highest = rsi[i];
        float lowest = rsi[i];
        for (int j = 1; j < kPeriod; ++j) {
          float v = rsi[i - j];
          if (v > highest)
            highest = v;
          if (v < lowest)
            lowest = v;
        }
        float denom = highest - lowest;
        kOut[i] = denom == 0.0f ? 0.0f : (rsi[i] - lowest) / denom * 100.0f;
        if (i >= rsiPeriod + kPeriod + dPeriod - 2) {
          float sum = 0.0f;
          for (int j = 0; j < dPeriod; ++j)
            sum += kOut[i - j];
          dOut[i] = sum / dPeriod;
        }
      }
    }
  }
}

StochRSI::StochRSI(int rsiPeriod, int kPeriod, int dPeriod)
    : rsiPeriod(rsiPeriod), kPeriod(kPeriod), dPeriod(dPeriod) {}

void StochRSI::calculate(const float *input, float *output,
                         int size, cudaStream_t stream) noexcept(false) {
  if (rsiPeriod <= 0 || kPeriod <= 0 || dPeriod <= 0 ||
      size <= rsiPeriod + kPeriod + dPeriod - 2) {
    throw std::invalid_argument("StochRSI: invalid parameters");
  }
  float *rsi = static_cast<float*>(DeviceBufferPool::instance().acquire(size * sizeof(float)));
  float *kOut = output;
  float *dOut = output + size;
  stochRsiKernel<<<1, 1, 0, stream>>>(input, rsi, kOut, dOut, rsiPeriod, kPeriod, dPeriod,
                           size);
  CUDA_CHECK(cudaGetLastError());
  DeviceBufferPool::instance().release(rsi);
}
