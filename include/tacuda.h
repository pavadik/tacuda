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
CTAPI_EXPORT ctStatus_t ct_wma(const float* host_input, float* host_output, int size, int period);
CTAPI_EXPORT ctStatus_t ct_momentum(const float* host_input, float* host_output, int size, int period);
CTAPI_EXPORT ctStatus_t ct_ema(const float* host_input, float* host_output, int size, int period);
CTAPI_EXPORT ctStatus_t ct_rsi(const float* host_input, float* host_output, int size, int period);
// MACD line only (EMA_fast - EMA_slow)
CTAPI_EXPORT ctStatus_t ct_macd_line(const float* host_input, float* host_output, int size,
                              int fastPeriod, int slowPeriod);
CTAPI_EXPORT ctStatus_t ct_bbands(const float* host_input,
                                  float* host_upper,
                                  float* host_middle,
                                  float* host_lower,
                                  int size,
                                  int period,
                                  float upperMul,
                                  float lowerMul);
CTAPI_EXPORT ctStatus_t ct_atr(const float* host_high,
                               const float* host_low,
                               const float* host_close,
                               float* host_output,
                               int size,
                               int period,
                               float initial);
CTAPI_EXPORT ctStatus_t ct_stochastic(const float* host_high,
                                      const float* host_low,
                                      const float* host_close,
                                      float* host_k,
                                      float* host_d,
                                      int size,
                                      int kPeriod,
                                      int dPeriod);
CTAPI_EXPORT ctStatus_t ct_cci(const float* host_high,
                               const float* host_low,
                               const float* host_close,
                               float* host_output,
                               int size,
                               int period);
CTAPI_EXPORT ctStatus_t ct_adx(const float* host_high,
                               const float* host_low,
                               const float* host_close,
                               float* host_output,
                               int size,
                               int period);
CTAPI_EXPORT ctStatus_t ct_obv(const float* host_price,
                               const float* host_volume,
                               float* host_output,
                               int size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TACUDA_H
