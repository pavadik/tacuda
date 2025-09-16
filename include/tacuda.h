#ifndef TACUDA_H
#define TACUDA_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>

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

typedef enum ctMaType {
  CT_MA_SMA = 0,
  CT_MA_EMA = 1,
} ctMaType_t;

// All APIs copy host->device->host internally for ease of binding.
CTAPI_EXPORT ctStatus_t ct_sma(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_ma(const float *host_input, float *host_output,
                              int size, int period, ctMaType_t type);
CTAPI_EXPORT ctStatus_t ct_wma(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_momentum(const float *host_input, float *host_output,
                                    int size, int period);
CTAPI_EXPORT ctStatus_t ct_change(const float *host_input, float *host_output,
                                  int size, int period);
CTAPI_EXPORT ctStatus_t ct_roc(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_rocp(const float *host_input, float *host_output,
                                int size, int period);
CTAPI_EXPORT ctStatus_t ct_rocr(const float *host_input, float *host_output,
                                int size, int period);
CTAPI_EXPORT ctStatus_t ct_rocr100(const float *host_input, float *host_output,
                                   int size, int period);
CTAPI_EXPORT ctStatus_t ct_ema(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_dema(const float *host_input, float *host_output,
                                int size, int period);
CTAPI_EXPORT ctStatus_t ct_tema(const float *host_input, float *host_output,
                                int size, int period);
CTAPI_EXPORT ctStatus_t ct_t3(const float *host_input, float *host_output,
                              int size, int period, float vFactor);
CTAPI_EXPORT ctStatus_t ct_trima(const float *host_input, float *host_output,
                                 int size, int period);
CTAPI_EXPORT ctStatus_t ct_trix(const float *host_input, float *host_output,
                                int size, int period);
CTAPI_EXPORT ctStatus_t ct_max(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_min(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_maxindex(const float *host_input, float *host_output,
                                    int size, int period);
CTAPI_EXPORT ctStatus_t ct_minindex(const float *host_input, float *host_output,
                                    int size, int period);
CTAPI_EXPORT ctStatus_t ct_minmax(const float *host_input, float *host_min,
                                  float *host_max, int size, int period);
CTAPI_EXPORT ctStatus_t ct_minmaxindex(const float *host_input,
                                       float *host_minidx, float *host_maxidx,
                                       int size, int period);
CTAPI_EXPORT ctStatus_t ct_stddev(const float *host_input, float *host_output,
                                  int size, int period);
CTAPI_EXPORT ctStatus_t ct_var(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_sum(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_rsi(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_kama(const float *host_input, float *host_output,
                                int size, int period, int fastPeriod,
                                int slowPeriod);
// MACD line only (EMA_fast - EMA_slow)
CTAPI_EXPORT ctStatus_t ct_macd_line(const float *host_input,
                                     float *host_output, int size,
                                     int fastPeriod, int slowPeriod);
CTAPI_EXPORT ctStatus_t ct_macd(const float *host_input, float *host_macd,
                                float *host_signal, float *host_hist, int size,
                                int fastPeriod, int slowPeriod,
                                int signalPeriod, ctMaType_t type);
CTAPI_EXPORT ctStatus_t ct_macdfix(const float *host_input, float *host_macd,
                                   float *host_signal, float *host_hist,
                                   int size, int signalPeriod);
CTAPI_EXPORT ctStatus_t ct_mama(const float *host_input, float *host_mama,
                                float *host_fama, int size, float fastLimit,
                                float slowLimit);
CTAPI_EXPORT ctStatus_t ct_apo(const float *host_input, float *host_output,
                               int size, int fastPeriod, int slowPeriod);
CTAPI_EXPORT ctStatus_t ct_ppo(const float *host_input, float *host_output,
                               int size, int fastPeriod, int slowPeriod);
CTAPI_EXPORT ctStatus_t ct_pvo(const float *host_volume, float *host_output,
                               int size, int fastPeriod, int slowPeriod);
// Device-pointer variants that operate directly on GPU memory.
CTAPI_EXPORT ctStatus_t ct_sma_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_ma_device(const float *device_input,
                                     float *device_output, int size,
                                     int period, ctMaType_t type,
                                     cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_wma_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_momentum_device(const float *device_input,
                                           float *device_output, int size,
                                           int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_change_device(const float *device_input,
                                         float *device_output, int size,
                                         int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_roc_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_rocp_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_rocr_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_rocr100_device(const float *device_input,
                                          float *device_output, int size,
                                          int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_ema_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_dema_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_tema_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_t3_device(const float *device_input,
                                     float *device_output, int size,
                                     int period, float vFactor,
                                     cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_trima_device(const float *device_input,
                                        float *device_output, int size,
                                        int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_trix_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_max_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_min_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_maxindex_device(const float *device_input,
                                           float *device_output, int size,
                                           int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_minindex_device(const float *device_input,
                                           float *device_output, int size,
                                           int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_stddev_device(const float *device_input,
                                         float *device_output, int size,
                                         int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_var_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_sum_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_rsi_device(const float *device_input,
                                      float *device_output, int size,
                                      int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_kama_device(const float *device_input,
                                       float *device_output, int size,
                                       int period, int fastPeriod,
                                       int slowPeriod, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_macd_line_device(const float *device_input,
                                            float *device_output, int size,
                                            int fastPeriod, int slowPeriod,
                                            cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_apo_device(const float *device_input,
                                      float *device_output, int size,
                                      int fastPeriod, int slowPeriod,
                                      cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_ppo_device(const float *device_input,
                                      float *device_output, int size,
                                      int fastPeriod, int slowPeriod,
                                      cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_pvo_device(const float *device_volume,
                                      float *device_output, int size,
                                      int fastPeriod, int slowPeriod,
                                      cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_bbands(const float *host_input, float *host_upper,
                                  float *host_middle, float *host_lower,
                                  int size, int period, float upperMul,
                                  float lowerMul);
CTAPI_EXPORT ctStatus_t ct_atr(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size, int period, float initial,
                               cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_natr(const float *host_high, const float *host_low,
                                const float *host_close, float *host_output,
                                int size, int period,
                                cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_trange(const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_stochastic(const float *host_high,
                                      const float *host_low,
                                      const float *host_close, float *host_k,
                                      float *host_d, int size, int kPeriod,
                                      int dPeriod, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_stochf(const float *host_high, const float *host_low,
                                  const float *host_close, float *host_k,
                                  float *host_d, int size, int kPeriod,
                                  int dPeriod, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_stochrsi(const float *host_input, float *host_k,
                                    float *host_d, int size, int rsiPeriod,
                                    int kPeriod, int dPeriod,
                                    cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_cci(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size, int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_adx(const float *host_high, const float *host_low,
                               const float *host_close, float *host_output,
                               int size, int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_adxr(const float *host_high, const float *host_low,
                                const float *host_close, float *host_output,
                                int size, int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_plus_dm(const float *host_high,
                                   const float *host_low, float *host_output,
                                   int size, int period);
CTAPI_EXPORT ctStatus_t ct_minus_dm(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size, int period);
CTAPI_EXPORT ctStatus_t ct_plus_di(const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_minus_di(const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_mfi(const float *host_high, const float *host_low,
                               const float *host_close,
                               const float *host_volume, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_obv(const float *host_price,
                               const float *host_volume, float *host_output,
                               int size);
CTAPI_EXPORT ctStatus_t ct_sar(const float *host_high, const float *host_low,
                               float *host_output, int size, float step,
                               float maxAcceleration);
CTAPI_EXPORT ctStatus_t ct_sarext(const float *host_high, const float *host_low,
                                  float *host_output, int size,
                                  float startValue, float offsetOnReverse,
                                  float accInitLong, float accLong,
                                  float accMaxLong, float accInitShort,
                                  float accShort, float accMaxShort);
CTAPI_EXPORT ctStatus_t ct_aroon(const float *host_high, const float *host_low,
                                 float *host_up, float *host_down,
                                 float *host_osc, int size, int upPeriod,
                                 int downPeriod);
CTAPI_EXPORT ctStatus_t ct_aroonosc(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size, int period);
CTAPI_EXPORT ctStatus_t ct_adosc(const float *host_high, const float *host_low,
                                 const float *host_close,
                                 const float *host_volume, float *host_output,
                                 int size, int shortPeriod, int longPeriod);
CTAPI_EXPORT ctStatus_t ct_ultosc(const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, int shortPeriod, int mediumPeriod,
                                  int longPeriod);
CTAPI_EXPORT ctStatus_t ct_ad(const float *host_high, const float *host_low,
                              const float *host_close, const float *host_volume,
                              float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_avgprice(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size);
CTAPI_EXPORT ctStatus_t ct_medprice(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size);
CTAPI_EXPORT ctStatus_t ct_typprice(const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size);
CTAPI_EXPORT ctStatus_t ct_wclprice(const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size);
CTAPI_EXPORT ctStatus_t ct_willr(const float *host_high, const float *host_low,
                                 const float *host_close, float *host_output,
                                 int size, int period);
CTAPI_EXPORT ctStatus_t ct_midpoint(const float *host_input, float *host_output,
                                    int size, int period);
CTAPI_EXPORT ctStatus_t ct_midprice(const float *host_high,
                                    const float *host_low, float *host_output,
                                    int size, int period);
CTAPI_EXPORT ctStatus_t ct_beta(const float *host_x, const float *host_y,
                                float *host_output, int size, int period);
CTAPI_EXPORT ctStatus_t ct_bop(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_doji(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size);
CTAPI_EXPORT ctStatus_t ct_cdl_hammer(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_inverted_hammer(const float *host_open,
                                               const float *host_high,
                                               const float *host_low,
                                               const float *host_close,
                                               float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_bullish_engulfing(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_bearish_engulfing(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_three_white_soldiers(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_abandoned_baby(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_advance_block(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_belt_hold(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_breakaway(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_two_crows(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_three_black_crows(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_three_inside(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_three_line_strike(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_three_stars_in_south(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_closing_marubozu(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_conceal_baby_swallow(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_counterattack(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_dark_cloud_cover(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_doji_star(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_dragonfly_doji(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_engulfing(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_doji_star(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_evening_star(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_gap_side_side_white(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_gravestone_doji(const float *host_open,
                                               const float *host_high,
                                               const float *host_low,
                                               const float *host_close,
                                               float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_hanging_man(const float *host_open,
                                           const float *host_high,
                                           const float *host_low,
                                           const float *host_close,
                                           float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_harami(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_harami_cross(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_high_wave(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_hikkake_mod(const float *host_open,
                                           const float *host_high,
                                           const float *host_low,
                                           const float *host_close,
                                           float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_homing_pigeon(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_identical_three_crows(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_in_neck(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_kicking_by_length(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_ladder_bottom(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_long_legged_doji(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_long_line(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_marubozu(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_matching_low(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_mat_hold(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_doji_star(const float *host_open,
                                                 const float *host_high,
                                                 const float *host_low,
                                                 const float *host_close,
                                                 float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_morning_star(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_on_neck(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_piercing(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_rickshaw_man(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_rise_fall3_methods(const float *host_open,
                                                  const float *host_high,
                                                  const float *host_low,
                                                  const float *host_close,
                                                  float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_separating_lines(const float *host_open,
                                                const float *host_high,
                                                const float *host_low,
                                                const float *host_close,
                                                float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_shooting_star(const float *host_open,
                                             const float *host_high,
                                             const float *host_low,
                                             const float *host_close,
                                             float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_short_line(const float *host_open,
                                          const float *host_high,
                                          const float *host_low,
                                          const float *host_close,
                                          float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_spinning_top(const float *host_open,
                                            const float *host_high,
                                            const float *host_low,
                                            const float *host_close,
                                            float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_stalled_pattern(const float *host_open,
                                               const float *host_high,
                                               const float *host_low,
                                               const float *host_close,
                                               float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_stick_sandwich(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_takuri(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_tasuki_gap(const float *host_open,
                                          const float *host_high,
                                          const float *host_low,
                                          const float *host_close,
                                          float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_thrusting(const float *host_open,
                                         const float *host_high,
                                         const float *host_low,
                                         const float *host_close,
                                         float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_tristar(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_unique_3_river(const float *host_open,
                                              const float *host_high,
                                              const float *host_low,
                                              const float *host_close,
                                              float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_upside_gap_2_crows(const float *host_open,
                                                  const float *host_high,
                                                  const float *host_low,
                                                  const float *host_close,
                                                  float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cdl_xside_gap_3_methods(
    const float *host_open, const float *host_high, const float *host_low,
    const float *host_close, float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_cmo(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_correl(const float *host_x, const float *host_y,
                                  float *host_output, int size, int period);
CTAPI_EXPORT ctStatus_t ct_dx(const float *host_high, const float *host_low,
                              const float *host_close, float *host_output,
                              int size, int period, cudaStream_t stream = 0);
CTAPI_EXPORT ctStatus_t ct_linearreg(const float *host_input,
                                     float *host_output, int size, int period);
CTAPI_EXPORT ctStatus_t ct_linearreg_slope(const float *host_input,
                                           float *host_output, int size,
                                           int period);
CTAPI_EXPORT ctStatus_t ct_linearreg_intercept(const float *host_input,
                                               float *host_output, int size,
                                               int period);
CTAPI_EXPORT ctStatus_t ct_linearreg_angle(const float *host_input,
                                           float *host_output, int size,
                                           int period);
CTAPI_EXPORT ctStatus_t ct_tsf(const float *host_input, float *host_output,
                               int size, int period);
CTAPI_EXPORT ctStatus_t ct_ht_dcperiod(const float *host_input,
                                       float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_ht_dcphase(const float *host_input,
                                      float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_ht_phasor(const float *host_input,
                                     float *host_inphase,
                                     float *host_quadrature, int size);
CTAPI_EXPORT ctStatus_t ct_ht_sine(const float *host_input, float *host_sine,
                                   float *host_leadsine, int size);
CTAPI_EXPORT ctStatus_t ct_ht_trendline(const float *host_input,
                                        float *host_output, int size);
CTAPI_EXPORT ctStatus_t ct_ht_trendmode(const float *host_input,
                                        float *host_output, int size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TACUDA_H
