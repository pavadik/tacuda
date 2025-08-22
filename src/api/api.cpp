#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tacuda.h"
#include <indicators/ADOSC.h>
#include <indicators/AD.h>
#include <indicators/ADXR.h>
#include <indicators/ADX.h>
#include <indicators/ATR.h>
#include <indicators/APO.h>
#include <indicators/Aroon.h>
#include <indicators/AroonOscillator.h>
#include <indicators/BBANDS.h>
#include <indicators/CCI.h>
#include <indicators/DEMA.h>
#include <indicators/EMA.h>
#include <indicators/AvgPrice.h>
#include <indicators/KAMA.h>
#include <indicators/MACD.h>
#include <indicators/MFI.h>
#include <indicators/Momentum.h>
#include <indicators/OBV.h>
#include <indicators/ROC.h>
#include <indicators/RSI.h>
#include <indicators/SAR.h>
#include <indicators/SMA.h>
#include <indicators/Stochastic.h>
#include <indicators/ULTOSC.h>
#include <indicators/TEMA.h>
#include <indicators/TRIX.h>
#include <indicators/WMA.h>
#include <indicators/MA.h>
#include <indicators/MAX.h>
#include <indicators/MAMA.h>
#include <indicators/MACDEXT.h>
#include <indicators/Beta.h>
#include <indicators/BOP.h>
#include <indicators/CMO.h>
#include <indicators/Correl.h>
#include <indicators/DX.h>
#include <indicators/HT_DCPERIOD.h>
#include <indicators/HT_DCPHASE.h>
#include <indicators/HT_PHASOR.h>
#include <indicators/HT_SINE.h>
#include <indicators/HT_TRENDMODE.h>
#include <indicators/LINEARREG.h>
#include <indicators/LINEARREG_ANGLE.h>
#include <indicators/LINEARREG_INTERCEPT.h>
#include <indicators/LINEARREG_SLOPE.h>
#include <utils/CudaUtils.h>

extern "C" {

struct CudaDeleter {
  void operator()(float *ptr) const noexcept {
    if (ptr)
      cudaFree(ptr);
  }
};
using DeviceBuffer = std::unique_ptr<float, CudaDeleter>;

static ctStatus_t run_indicator(Indicator &ind, const float *h_in, float *h_out,
                                int size, int outMultiple = 1) {
  DeviceBuffer d_in{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_in.get(), h_in, size * sizeof(float),
                   cudaMemcpyHostToDevice);
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

ctStatus_t ct_sma(const float *host_input, float *host_output, int size,
                  int period) {
  SMA sma(period);
  return run_indicator(sma, host_input, host_output, size);
}

ctStatus_t ct_ma(const float *host_input, float *host_output, int size,
                 int period, ctMaType_t type) {
  MA ma(period, static_cast<MAType>(type));
  return run_indicator(ma, host_input, host_output, size);
}

ctStatus_t ct_wma(const float *host_input, float *host_output, int size,
                  int period) {
  WMA wma(period);
  return run_indicator(wma, host_input, host_output, size);
}

ctStatus_t ct_momentum(const float *host_input, float *host_output, int size,
                       int period) {
  Momentum mom(period);
  return run_indicator(mom, host_input, host_output, size);
}

ctStatus_t ct_roc(const float *host_input, float *host_output, int size,
                  int period) {
  ROC roc(period);
  return run_indicator(roc, host_input, host_output, size);
}

ctStatus_t ct_ema(const float *host_input, float *host_output, int size,
                  int period) {
  EMA ema(period);
  return run_indicator(ema, host_input, host_output, size);
}

ctStatus_t ct_dema(const float *host_input, float *host_output, int size,
                   int period) {
  DEMA dema(period);
  return run_indicator(dema, host_input, host_output, size);
}

ctStatus_t ct_tema(const float *host_input, float *host_output, int size,
                   int period) {
  TEMA tema(period);
  return run_indicator(tema, host_input, host_output, size);
}

ctStatus_t ct_trix(const float *host_input, float *host_output, int size,
                   int period) {
  TRIX trix(period);
  return run_indicator(trix, host_input, host_output, size);
}

ctStatus_t ct_max(const float *host_input, float *host_output, int size,
                  int period) {
  MAX mx(period);
  return run_indicator(mx, host_input, host_output, size);
}

ctStatus_t ct_rsi(const float *host_input, float *host_output, int size,
                  int period) {
  RSI rsi(period);
  return run_indicator(rsi, host_input, host_output, size);
}

ctStatus_t ct_kama(const float *host_input, float *host_output, int size,
                   int period, int fastPeriod, int slowPeriod) {
  KAMA kama(period, fastPeriod, slowPeriod);
  return run_indicator(kama, host_input, host_output, size);
}

ctStatus_t ct_macd_line(const float *host_input, float *host_output, int size,
                        int fastPeriod, int slowPeriod) {
  MACD macd(fastPeriod, slowPeriod);
  return run_indicator(macd, host_input, host_output, size);
}

ctStatus_t ct_macd(const float *host_input, float *host_macd,
                   float *host_signal, float *host_hist, int size,
                   int fastPeriod, int slowPeriod, int signalPeriod,
                   ctMaType_t type) {
  MACDEXT macd(fastPeriod, slowPeriod, signalPeriod,
               static_cast<MAType>(type));
  std::vector<float> tmp(3 * size);
  ctStatus_t rc = run_indicator(macd, host_input, tmp.data(), size, 3);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_macd, tmp.data(), size * sizeof(float));
  std::memcpy(host_signal, tmp.data() + size, size * sizeof(float));
  std::memcpy(host_hist, tmp.data() + 2 * size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_mama(const float *host_input, float *host_mama,
                   float *host_fama, int size,
                   float fastLimit, float slowLimit) {
  MAMA ma(fastLimit, slowLimit);
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(ma, host_input, tmp.data(), size, 2);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_mama, tmp.data(), size * sizeof(float));
  std::memcpy(host_fama, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_apo(const float *host_input, float *host_output, int size,
                  int fastPeriod, int slowPeriod) {
  APO apo(fastPeriod, slowPeriod);
  return run_indicator(apo, host_input, host_output, size);
}

ctStatus_t ct_bbands(const float *host_input, float *host_upper,
                     float *host_middle, float *host_lower, int size,
                     int period, float upperMul, float lowerMul) {
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

ctStatus_t ct_atr(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period, float initial) {
  ATR atr(period, initial);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    atr.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_stochastic(const float *host_high, const float *host_low,
                         const float *host_close, float *host_k, float *host_d,
                         int size, int kPeriod, int dPeriod) {
  Stochastic stoch(kPeriod, dPeriod);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    stoch.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(),
                    size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  std::vector<float> tmpHost(2 * size);
  err = cudaMemcpy(tmpHost.data(), d_out.get(), 2 * size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  std::memcpy(host_k, tmpHost.data(), size * sizeof(float));
  std::memcpy(host_d, tmpHost.data() + size, size * sizeof(float));

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_cci(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period) {
  CCI cci(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    cci.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_adx(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period) {
  ADX adx(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    adx.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_adxr(const float *host_high, const float *host_low,
                   const float *host_close, float *host_output, int size,
                   int period) {
  ADXR adxr(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    adxr.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_sar(const float *host_high, const float *host_low,
                  float *host_output, int size, float step,
                  float maxAcceleration) {
  SAR sar(step, maxAcceleration);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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
  d_out.reset(tmp);

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    sar.calculate(d_high.get(), d_low.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_aroon(const float *host_high, const float *host_low,
                    float *host_up, float *host_down, float *host_osc, int size,
                    int upPeriod, int downPeriod) {
  Aroon aroon(upPeriod, downPeriod);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMalloc(&tmp, 3 * size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    aroon.calculate(d_high.get(), d_low.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  std::vector<float> tmpHost(3 * size);
  err = cudaMemcpy(tmpHost.data(), d_out.get(), 3 * size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  std::memcpy(host_up, tmpHost.data(), size * sizeof(float));
  std::memcpy(host_down, tmpHost.data() + size, size * sizeof(float));
  std::memcpy(host_osc, tmpHost.data() + 2 * size, size * sizeof(float));

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_aroonosc(const float *host_high, const float *host_low,
                       float *host_output, int size, int period) {
  AroonOscillator ind(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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
  d_out.reset(tmp);

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_high.get(), d_low.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_adosc(const float *host_high, const float *host_low,
                    const float *host_close, const float *host_volume,
                    float *host_output, int size, int shortPeriod,
                    int longPeriod) {
  ADOSC adosc(shortPeriod, longPeriod);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_vol{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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
  d_vol.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_vol.get(), host_volume, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    adosc.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                    d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ad(const float *host_high, const float *host_low,
                 const float *host_close, const float *host_volume,
                 float *host_output, int size) {
  AD ad;
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_vol{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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
  d_vol.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_vol.get(), host_volume, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ad.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                 d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_avgprice(const float *host_open, const float *host_high,
                       const float *host_low, const float *host_close,
                       float *host_output, int size) {
  AvgPrice ap;
  DeviceBuffer d_open{nullptr}, d_high{nullptr}, d_low{nullptr},
      d_close{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

  cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_open.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
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

  err = cudaMemcpy(d_open.get(), host_open, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ap.calculate(d_open.get(), d_high.get(), d_low.get(), d_close.get(),
                 d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ultosc(const float *host_high, const float *host_low,
                     const float *host_close, float *host_output, int size,
                     int shortPeriod, int mediumPeriod, int longPeriod) {
  ULTOSC ultosc(shortPeriod, mediumPeriod, longPeriod);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ultosc.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(),
                     size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_mfi(const float *host_high, const float *host_low,
                  const float *host_close, const float *host_volume,
                  float *host_output, int size, int period) {
  MFI mfi(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_vol{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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
  d_vol.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  err = cudaMemcpy(d_high.get(), host_high, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_low.get(), host_low, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_close.get(), host_close, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_vol.get(), host_volume, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mfi.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                  d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_obv(const float *host_price, const float *host_volume,
                  float *host_output, int size) {
  OBV obv;
  DeviceBuffer d_price{nullptr}, d_volume{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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

  err = cudaMemcpy(d_price.get(), host_price, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_volume.get(), host_volume, size * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    obv.calculate(d_price.get(), d_volume.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_beta(const float *host_x, const float *host_y,
                   float *host_output, int size, int period) {
  Beta beta(period);
  DeviceBuffer d_x{nullptr}, d_y{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

  cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_x.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_y.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  err = cudaMemcpy(d_x.get(), host_x, size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_y.get(), host_y, size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    beta.calculate(d_x.get(), d_y.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ht_dcperiod(const float *host_input, float *host_output, int size) {
  HT_DCPERIOD ind;
  return run_indicator(ind, host_input, host_output, size);
}

ctStatus_t ct_ht_dcphase(const float *host_input, float *host_output, int size) {
  HT_DCPHASE ind;
  return run_indicator(ind, host_input, host_output, size);
}

ctStatus_t ct_ht_phasor(const float *host_input, float *host_inphase,
                        float *host_quadrature, int size) {
  HT_PHASOR ind;
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(ind, host_input, tmp.data(), size, 2);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_inphase, tmp.data(), size * sizeof(float));
  std::memcpy(host_quadrature, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ht_sine(const float *host_input, float *host_sine,
                      float *host_leadsine, int size) {
  HT_SINE ind;
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(ind, host_input, tmp.data(), size, 2);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_sine, tmp.data(), size * sizeof(float));
  std::memcpy(host_leadsine, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ht_trendmode(const float *host_input, float *host_output, int size) {
  HT_TRENDMODE ind;
  return run_indicator(ind, host_input, host_output, size);
}

ctStatus_t ct_linearreg(const float *host_input, float *host_output, int size,
                        int period) {
  LINEARREG ind(period);
  return run_indicator(ind, host_input, host_output, size);
}

ctStatus_t ct_linearreg_slope(const float *host_input, float *host_output,
                              int size, int period) {
  LINEARREG_SLOPE ind(period);
  return run_indicator(ind, host_input, host_output, size);
}

ctStatus_t ct_linearreg_intercept(const float *host_input, float *host_output,
                                  int size, int period) {
  LINEARREG_INTERCEPT ind(period);
  return run_indicator(ind, host_input, host_output, size);
}

ctStatus_t ct_linearreg_angle(const float *host_input, float *host_output,
                              int size, int period) {
  LINEARREG_ANGLE ind(period);
  return run_indicator(ind, host_input, host_output, size);
}

ctStatus_t ct_bop(const float *host_open, const float *host_high,
                  const float *host_low, const float *host_close,
                  float *host_output, int size) {
  BOP bop;
  DeviceBuffer d_open{nullptr}, d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  float *tmp = nullptr;

  cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_open.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
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

  err = cudaMemcpy(d_open.get(), host_open, size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
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
    bop.calculate(d_open.get(), d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_cmo(const float *host_input, float *host_output, int size,
                  int period) {
  CMO cmo(period);
  return run_indicator(cmo, host_input, host_output, size);
}

ctStatus_t ct_correl(const float *host_x, const float *host_y,
                     float *host_output, int size, int period) {
  Correl correl(period);
  DeviceBuffer d_x{nullptr}, d_y{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

  cudaError_t err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_x.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_y.reset(tmp);

  err = cudaMalloc(&tmp, size * sizeof(float));
  if (err != cudaSuccess) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  err = cudaMemcpy(d_x.get(), host_x, size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpy(d_y.get(), host_y, size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    correl.calculate(d_x.get(), d_y.get(), d_out.get(), size);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpy(host_output, d_out.get(), size * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_dx(const float *host_high, const float *host_low,
                 const float *host_close, float *host_output, int size,
                 int period) {
  DX dx(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

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
    dx.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size);
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
