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

// All APIs copy host->device->host internally for ease of binding.
// Returns 0 on success, non-zero on failure.
CTAPI_EXPORT int ct_sma(const float* host_input, float* host_output, int size, int period);
CTAPI_EXPORT int ct_momentum(const float* host_input, float* host_output, int size, int period);
// MACD line only (EMA_fast - EMA_slow)
CTAPI_EXPORT int ct_macd_line(const float* host_input, float* host_output, int size,
                              int fastPeriod, int slowPeriod, int signalPeriod);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TACUDA_H
