#pragma once

#include <cuda_runtime.h>

namespace tacuda::indicators::detail {

void computeEmaDevice(const float* input,
                      float* output,
                      int size,
                      int period,
                      cudaStream_t stream);

}

