#ifndef TACUDA_H
#define TACUDA_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #define CTAPI_EXPORT __declspec(dllexport)
#else
  #define CTAPI_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ctStatus {
    CT_STATUS_SUCCESS = 0,
    CT_STATUS_ALLOC_FAILED = 1,
    CT_STATUS_COPY_FAILED = 2,
    CT_STATUS_KERNEL_FAILED = 3,
} ctStatus_t;

// All APIs copy host->device->host internally for ease of binding.
CTAPI_EXPORT ctStatus_t ct_sma(const float* host_input, float* host_output, int size, int period);
CTAPI_EXPORT ctStatus_t ct_momentum(const float* host_input, float* host_output, int size, int period);
// MACD: compute line, signal and histogram arrays
CTAPI_EXPORT ctStatus_t ct_macd(const float* host_input,
                                float* macdOut,
                                float* signalOut,
                                float* histOut,
                                int size,
                                int fastPeriod, int slowPeriod, int signalPeriod);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TACUDA_H
