#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tacuda.h"
#include <indicators/AD.h>
#include <indicators/ACCBANDS.h>
#include <indicators/ADOSC.h>
#include <indicators/ADX.h>
#include <indicators/ADXR.h>
#include <indicators/AVGDEV.h>
#include <indicators/APO.h>
#include <indicators/ATR.h>
#include <indicators/AbandonedBaby.h>
#include <indicators/AdvanceBlock.h>
#include <indicators/Aroon.h>
#include <indicators/AroonOscillator.h>
#include <indicators/AvgPrice.h>
#include <indicators/BBANDS.h>
#include <indicators/BOP.h>
#include <indicators/BearishEngulfing.h>
#include <indicators/BeltHold.h>
#include <indicators/Beta.h>
#include <indicators/Breakaway.h>
#include <indicators/BullishEngulfing.h>
#include <indicators/CCI.h>
#include <indicators/CMO.h>
#include <indicators/Change.h>
#include <indicators/ClosingMarubozu.h>
#include <indicators/ConcealBabySwallow.h>
#include <indicators/Correl.h>
#include <indicators/CounterAttack.h>
#include <indicators/CDL3OUTSIDE.h>
#include <indicators/DEMA.h>
#include <indicators/DX.h>
#include <indicators/DarkCloudCover.h>
#include <indicators/Doji.h>
#include <indicators/IMI.h>
#include <indicators/DojiStar.h>
#include <indicators/DragonflyDoji.h>
#include <indicators/EMA.h>
#include <indicators/Engulfing.h>
#include <indicators/EveningDojiStar.h>
#include <indicators/EveningStar.h>
#include <indicators/GapSideSideWhite.h>
#include <indicators/GravestoneDoji.h>
#include <indicators/HT_DCPERIOD.h>
#include <indicators/HT_DCPHASE.h>
#include <indicators/HT_PHASOR.h>
#include <indicators/HT_SINE.h>
#include <indicators/HT_TRENDLINE.h>
#include <indicators/HT_TRENDMODE.h>
#include <indicators/Hammer.h>
#include <indicators/HangingMan.h>
#include <indicators/Harami.h>
#include <indicators/HaramiCross.h>
#include <indicators/HighWave.h>
#include <indicators/Hikkake.h>
#include <indicators/HikkakeMod.h>
#include <indicators/HomingPigeon.h>
#include <indicators/IdenticalThreeCrows.h>
#include <indicators/InNeck.h>
#include <indicators/InvertedHammer.h>
#include <indicators/KAMA.h>
#include <indicators/Kicking.h>
#include <indicators/KickingByLength.h>
#include <indicators/LINEARREG.h>
#include <indicators/LINEARREG_ANGLE.h>
#include <indicators/LINEARREG_INTERCEPT.h>
#include <indicators/LINEARREG_SLOPE.h>
#include <indicators/LadderBottom.h>
#include <indicators/LongLeggedDoji.h>
#include <indicators/LongLine.h>
#include <indicators/MA.h>
#include <indicators/MAVP.h>
#include <indicators/MACD.h>
#include <indicators/MACDEXT.h>
#include <indicators/MACDFIX.h>
#include <indicators/MAMA.h>
#include <indicators/MAX.h>
#include <indicators/MAXINDEX.h>
#include <indicators/MFI.h>
#include <indicators/MIN.h>
#include <indicators/MININDEX.h>
#include <indicators/MINMAX.h>
#include <indicators/MINMAXINDEX.h>
#include <indicators/Marubozu.h>
#include <indicators/MatHold.h>
#include <indicators/MatchingLow.h>
#include <indicators/MedPrice.h>
#include <indicators/MIDPOINT.h>
#include <indicators/MIDPRICE.h>
#include <indicators/MinusDI.h>
#include <indicators/MinusDM.h>
#include <indicators/Momentum.h>
#include <indicators/NVI.h>
#include <indicators/MorningDojiStar.h>
#include <indicators/MorningStar.h>
#include <indicators/NATR.h>
#include <indicators/PVI.h>
#include <indicators/OBV.h>
#include <indicators/OnNeck.h>
#include <indicators/PPO.h>
#include <indicators/PVO.h>
#include <indicators/Piercing.h>
#include <indicators/PlusDI.h>
#include <indicators/PlusDM.h>
#include <indicators/ROC.h>
#include <indicators/ROCP.h>
#include <indicators/ROCR.h>
#include <indicators/ROCR100.h>
#include <indicators/RSI.h>
#include <indicators/RickshawMan.h>
#include <indicators/RiseFall3Methods.h>
#include <indicators/SAR.h>
#include <indicators/SAREXT.h>
#include <indicators/SMA.h>
#include <indicators/SUM.h>
#include <indicators/SeparatingLines.h>
#include <indicators/ShootingStar.h>
#include <indicators/ShortLine.h>
#include <indicators/SpinningTop.h>
#include <indicators/StalledPattern.h>
#include <indicators/StdDev.h>
#include <indicators/StickSandwich.h>
#include <indicators/StochRSI.h>
#include <indicators/Stochastic.h>
#include <indicators/StochasticFast.h>
#include <indicators/T3.h>
#include <indicators/TEMA.h>
#include <indicators/TRANGE.h>
#include <indicators/TRIMA.h>
#include <indicators/TRIX.h>
#include <indicators/TSF.h>
#include <indicators/Takuri.h>
#include <indicators/TasukiGap.h>
#include <indicators/ThreeBlackCrows.h>
#include <indicators/ThreeInside.h>
#include <indicators/ThreeLineStrike.h>
#include <indicators/ThreeStarsInSouth.h>
#include <indicators/ThreeWhiteSoldiers.h>
#include <indicators/Thrusting.h>
#include <indicators/Tristar.h>
#include <indicators/TwoCrows.h>
#include <indicators/TypPrice.h>
#include <indicators/ULTOSC.h>
#include <indicators/Unique3River.h>
#include <indicators/UpsideGap2Crows.h>
#include <indicators/VAR.h>
#include <indicators/WMA.h>
#include <indicators/WclPrice.h>
#include <indicators/WILLR.h>
#include <indicators/XSideGap3Methods.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

struct PooledDeleter {
  void operator()(float *ptr) const noexcept {
    DeviceBufferPool::instance().release(ptr);
  }
};
using DeviceBuffer = std::unique_ptr<float, PooledDeleter>;

static bool acquire_buffer(DeviceBuffer &buffer, size_t elements) {
  float *tmp = nullptr;
  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(elements * sizeof(float)));
  } catch (...) {
    return false;
  }
  buffer.reset(tmp);
  return true;
}

static ctStatus_t run_indicator(tacuda::Indicator &ind, const float *h_in, float *h_out,
                                int size, int outMultiple = 1,
                                cudaStream_t stream = 0) {
  DeviceBuffer d_in{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_in.reset(tmp);

  try {
    tmp = static_cast<float *>(DeviceBufferPool::instance().acquire(
        size * outMultiple * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  cudaError_t err = cudaMemcpyAsync(d_in.get(), h_in, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_in.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(h_out, d_out.get(),
                        size * outMultiple * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

template <typename Indicator>
static ctStatus_t run_indicator_device(Indicator &ind, const float *d_in,
                                       float *d_out, int size,
                                       cudaStream_t stream = 0) {
  try {
    ind.calculate(d_in, d_out, size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

template <typename T>
static ctStatus_t run_ohlc_indicator(T &ind, const float *h_open,
                                     const float *h_high, const float *h_low,
                                     const float *h_close, float *h_out,
                                     int size, int outMultiple = 1,
                                     cudaStream_t stream = 0) {
  DeviceBuffer d_open{nullptr}, d_high{nullptr}, d_low{nullptr},
      d_close{nullptr}, d_out{nullptr};
  float *tmp = nullptr;

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_open.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_high.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_low.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_close.reset(tmp);

  try {
    tmp = static_cast<float *>(DeviceBufferPool::instance().acquire(
        size * outMultiple * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  cudaError_t err = cudaMemcpyAsync(d_open.get(), h_open, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_high.get(), h_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), h_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), h_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_open.get(), d_high.get(), d_low.get(), d_close.get(),
                  d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(h_out, d_out.get(),
                        size * outMultiple * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

template <typename T>
static ctStatus_t run_ohlc_indicator(T &ind, const float *h_high,
                                     const float *h_low, const float *h_close,
                                     float *h_out, int size,
                                     int outMultiple = 1,
                                     cudaStream_t stream = 0) {
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  float *tmp = nullptr;

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_high.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_low.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_close.reset(tmp);

  try {
    tmp = static_cast<float *>(DeviceBufferPool::instance().acquire(
        size * outMultiple * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  cudaError_t err = cudaMemcpyAsync(d_high.get(), h_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), h_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), h_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size,
                  stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(h_out, d_out.get(),
                        size * outMultiple * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

extern "C" {

ctStatus_t ct_sma(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::SMA sma(period);
  return run_indicator(sma, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_ma(const float *host_input, float *host_output, int size,
                 int period, ctMaType_t type, cudaStream_t stream) {
  tacuda::MA ma(period, static_cast<tacuda::MAType>(type));
  return run_indicator(ma, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_mavp(const float *host_input, const float *host_periods,
                   float *host_output, int size, int minPeriod, int maxPeriod,
                   ctMaType_t type, cudaStream_t stream) {
  tacuda::MAVP mavp(minPeriod, maxPeriod, static_cast<tacuda::MAType>(type));

  DeviceBuffer d_values{nullptr}, d_periods{nullptr}, d_out{nullptr};
  if (!acquire_buffer(d_values, size) || !acquire_buffer(d_periods, size) ||
      !acquire_buffer(d_out, size)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err =
      cudaMemcpyAsync(d_values.get(), host_input, size * sizeof(float),
                      cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaMemcpyAsync(d_periods.get(), host_periods, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mavp.calculate(d_values.get(), d_periods.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_wma(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::WMA wma(period);
  return run_indicator(wma, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_momentum(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  tacuda::Momentum mom(period);
  return run_indicator(mom, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_change(const float *host_input, float *host_output, int size,
                     int period, cudaStream_t stream) {
  tacuda::Change ch(period);
  return run_indicator(ch, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_roc(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::ROC roc(period);
  return run_indicator(roc, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_rocp(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  tacuda::ROCP rocp(period);
  return run_indicator(rocp, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_rocr(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  tacuda::ROCR rocr(period);
  return run_indicator(rocr, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_rocr100(const float *host_input, float *host_output, int size,
                      int period, cudaStream_t stream) {
  tacuda::ROCR100 rocr100(period);
  return run_indicator(rocr100, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_ema(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::EMA ema(period);
  return run_indicator(ema, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_dema(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  tacuda::DEMA dema(period);
  return run_indicator(dema, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_tema(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  tacuda::TEMA tema(period);
  return run_indicator(tema, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_t3(const float *host_input, float *host_output, int size,
                 int period, float vFactor, cudaStream_t stream) {
  tacuda::T3 t3(period, vFactor);
  return run_indicator(t3, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_trima(const float *host_input, float *host_output, int size,
                    int period, cudaStream_t stream) {
  tacuda::TRIMA trima(period);
  return run_indicator(trima, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_trix(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  tacuda::TRIX trix(period);
  return run_indicator(trix, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_max(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::MAX mx(period);
  return run_indicator(mx, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_min(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::MIN mn(period);
  return run_indicator(mn, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_maxindex(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  tacuda::MAXINDEX mi(period);
  return run_indicator(mi, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_minindex(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  tacuda::MININDEX mi(period);
  return run_indicator(mi, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_minmax(const float *host_input, float *host_min, float *host_max,
                     int size, int period, cudaStream_t stream) {
  tacuda::MINMAX mm(period);
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(mm, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_min, tmp.data(), size * sizeof(float));
  std::memcpy(host_max, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_minmaxindex(const float *host_input, float *host_minidx,
                          float *host_maxidx, int size, int period, cudaStream_t stream) {
  tacuda::MINMAXINDEX mm(period);
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(mm, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_minidx, tmp.data(), size * sizeof(float));
  std::memcpy(host_maxidx, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_stddev(const float *host_input, float *host_output, int size,
                     int period, cudaStream_t stream) {
  tacuda::StdDev sd(period);
  return run_indicator(sd, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_avgdev(const float *host_input, float *host_output, int size,
                     int period, cudaStream_t stream) {
  tacuda::AVGDEV avg(period);
  return run_indicator(avg, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_var(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::VAR vr(period);
  return run_indicator(vr, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_sum(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::SUM sum(period);
  return run_indicator(sum, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_rsi(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::RSI rsi(period);
  return run_indicator(rsi, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_kama(const float *host_input, float *host_output, int size,
                   int period, int fastPeriod, int slowPeriod, cudaStream_t stream) {
  tacuda::KAMA kama(period, fastPeriod, slowPeriod);
  return run_indicator(kama, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_macd_line(const float *host_input, float *host_output, int size,
                        int fastPeriod, int slowPeriod, cudaStream_t stream) {
  tacuda::MACD macd(fastPeriod, slowPeriod);
  return run_indicator(macd, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_macd(const float *host_input, float *host_macd,
                   float *host_signal, float *host_hist, int size,
                   int fastPeriod, int slowPeriod, int signalPeriod,
                   ctMaType_t type, cudaStream_t stream) {
  tacuda::MACDEXT macd(fastPeriod, slowPeriod, signalPeriod, static_cast<tacuda::MAType>(type));
  std::vector<float> tmp(3 * size);
  ctStatus_t rc = run_indicator(macd, host_input, tmp.data(), size, 3, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_macd, tmp.data(), size * sizeof(float));
  std::memcpy(host_signal, tmp.data() + size, size * sizeof(float));
  std::memcpy(host_hist, tmp.data() + 2 * size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_macdfix(const float *host_input, float *host_macd,
                      float *host_signal, float *host_hist, int size,
                      int signalPeriod, cudaStream_t stream) {
  tacuda::MACDFIX macd(signalPeriod);
  std::vector<float> tmp(3 * size);
  ctStatus_t rc = run_indicator(macd, host_input, tmp.data(), size, 3, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_macd, tmp.data(), size * sizeof(float));
  std::memcpy(host_signal, tmp.data() + size, size * sizeof(float));
  std::memcpy(host_hist, tmp.data() + 2 * size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_mama(const float *host_input, float *host_mama, float *host_fama,
                   int size, float fastLimit, float slowLimit, cudaStream_t stream) {
  tacuda::MAMA ma(fastLimit, slowLimit);
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(ma, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_mama, tmp.data(), size * sizeof(float));
  std::memcpy(host_fama, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_apo(const float *host_input, float *host_output, int size,
                  int fastPeriod, int slowPeriod, cudaStream_t stream) {
  tacuda::APO apo(fastPeriod, slowPeriod);
  return run_indicator(apo, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_ppo(const float *host_input, float *host_output, int size,
                  int fastPeriod, int slowPeriod, cudaStream_t stream) {
  tacuda::PPO ppo(fastPeriod, slowPeriod);
  return run_indicator(ppo, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_pvo(const float *host_volume, float *host_output, int size,
                  int fastPeriod, int slowPeriod, cudaStream_t stream) {
  tacuda::PVO pvo(fastPeriod, slowPeriod);
  return run_indicator(pvo, host_volume, host_output, size, 1, stream);
}

ctStatus_t ct_sma_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::SMA sma(period);
  return run_indicator_device(sma, device_input, device_output, size, stream);
}

ctStatus_t ct_ma_device(const float *device_input, float *device_output,
                        int size, int period, ctMaType_t type,
                        cudaStream_t stream) {
  tacuda::MA ma(period, static_cast<tacuda::MAType>(type));
  return run_indicator_device(ma, device_input, device_output, size, stream);
}

ctStatus_t ct_wma_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::WMA wma(period);
  return run_indicator_device(wma, device_input, device_output, size, stream);
}

ctStatus_t ct_momentum_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  tacuda::Momentum mom(period);
  return run_indicator_device(mom, device_input, device_output, size, stream);
}

ctStatus_t ct_change_device(const float *device_input, float *device_output,
                            int size, int period, cudaStream_t stream) {
  tacuda::Change ch(period);
  return run_indicator_device(ch, device_input, device_output, size, stream);
}

ctStatus_t ct_roc_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::ROC roc(period);
  return run_indicator_device(roc, device_input, device_output, size, stream);
}

ctStatus_t ct_rocp_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  tacuda::ROCP rocp(period);
  return run_indicator_device(rocp, device_input, device_output, size, stream);
}

ctStatus_t ct_rocr_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  tacuda::ROCR rocr(period);
  return run_indicator_device(rocr, device_input, device_output, size, stream);
}

ctStatus_t ct_rocr100_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  tacuda::ROCR100 rocr100(period);
  return run_indicator_device(rocr100, device_input, device_output, size,
                              stream);
}

ctStatus_t ct_ema_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::EMA ema(period);
  return run_indicator_device(ema, device_input, device_output, size, stream);
}

ctStatus_t ct_dema_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  tacuda::DEMA dema(period);
  return run_indicator_device(dema, device_input, device_output, size, stream);
}

ctStatus_t ct_tema_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  tacuda::TEMA tema(period);
  return run_indicator_device(tema, device_input, device_output, size, stream);
}

ctStatus_t ct_t3_device(const float *device_input, float *device_output,
                        int size, int period, float vFactor,
                        cudaStream_t stream) {
  tacuda::T3 t3(period, vFactor);
  return run_indicator_device(t3, device_input, device_output, size, stream);
}

ctStatus_t ct_trima_device(const float *device_input, float *device_output,
                           int size, int period, cudaStream_t stream) {
  tacuda::TRIMA trima(period);
  return run_indicator_device(trima, device_input, device_output, size, stream);
}

ctStatus_t ct_trix_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  tacuda::TRIX trix(period);
  return run_indicator_device(trix, device_input, device_output, size, stream);
}

ctStatus_t ct_max_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::MAX mx(period);
  return run_indicator_device(mx, device_input, device_output, size, stream);
}

ctStatus_t ct_min_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::MIN mn(period);
  return run_indicator_device(mn, device_input, device_output, size, stream);
}

ctStatus_t ct_maxindex_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  tacuda::MAXINDEX mi(period);
  return run_indicator_device(mi, device_input, device_output, size, stream);
}

ctStatus_t ct_minindex_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  tacuda::MININDEX mi(period);
  return run_indicator_device(mi, device_input, device_output, size, stream);
}

ctStatus_t ct_stddev_device(const float *device_input, float *device_output,
                            int size, int period, cudaStream_t stream) {
  tacuda::StdDev sd(period);
  return run_indicator_device(sd, device_input, device_output, size, stream);
}

ctStatus_t ct_var_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::VAR vr(period);
  return run_indicator_device(vr, device_input, device_output, size, stream);
}

ctStatus_t ct_sum_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::SUM sum(period);
  return run_indicator_device(sum, device_input, device_output, size, stream);
}

ctStatus_t ct_rsi_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  tacuda::RSI rsi(period);
  return run_indicator_device(rsi, device_input, device_output, size, stream);
}

ctStatus_t ct_kama_device(const float *device_input, float *device_output,
                          int size, int period, int fastPeriod, int slowPeriod,
                          cudaStream_t stream) {
  tacuda::KAMA kama(period, fastPeriod, slowPeriod);
  return run_indicator_device(kama, device_input, device_output, size, stream);
}

ctStatus_t ct_macd_line_device(const float *device_input,
                               float *device_output, int size, int fastPeriod,
                               int slowPeriod, cudaStream_t stream) {
  tacuda::MACD macd(fastPeriod, slowPeriod);
  return run_indicator_device(macd, device_input, device_output, size, stream);
}

ctStatus_t ct_apo_device(const float *device_input, float *device_output,
                         int size, int fastPeriod, int slowPeriod,
                         cudaStream_t stream) {
  tacuda::APO apo(fastPeriod, slowPeriod);
  return run_indicator_device(apo, device_input, device_output, size, stream);
}

ctStatus_t ct_ppo_device(const float *device_input, float *device_output,
                         int size, int fastPeriod, int slowPeriod,
                         cudaStream_t stream) {
  tacuda::PPO ppo(fastPeriod, slowPeriod);
  return run_indicator_device(ppo, device_input, device_output, size, stream);
}

ctStatus_t ct_pvo_device(const float *device_volume, float *device_output,
                         int size, int fastPeriod, int slowPeriod,
                         cudaStream_t stream) {
  tacuda::PVO pvo(fastPeriod, slowPeriod);
  return run_indicator_device(pvo, device_volume, device_output, size, stream);
}

ctStatus_t ct_accbands(const float *host_high, const float *host_low,
                       const float *host_close, float *host_upper,
                       float *host_middle, float *host_lower, int size,
                       int period, cudaStream_t stream) {
  tacuda::ACCBANDS acc(period);
  std::vector<float> tmp(3 * size);
  ctStatus_t rc = run_ohlc_indicator(acc, host_high, host_low, host_close,
                                     tmp.data(), size, 3, stream);
  if (rc != CT_STATUS_SUCCESS) {
    return rc;
  }
  std::memcpy(host_upper, tmp.data(), size * sizeof(float));
  std::memcpy(host_middle, tmp.data() + size, size * sizeof(float));
  std::memcpy(host_lower, tmp.data() + 2 * size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_bbands(const float *host_input, float *host_upper,
                     float *host_middle, float *host_lower, int size,
                     int period, float upperMul, float lowerMul, cudaStream_t stream) {
  tacuda::BBANDS bb(period, upperMul, lowerMul);
  std::vector<float> tmp(3 * size);
  ctStatus_t rc = run_indicator(bb, host_input, tmp.data(), size, 3, stream);
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
                  int period, float initial, cudaStream_t stream) {
  tacuda::ATR atr(period, initial);
  return run_ohlc_indicator(atr, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_natr(const float *host_high, const float *host_low,
                   const float *host_close, float *host_output, int size,
                   int period, cudaStream_t stream) {
  tacuda::NATR natr(period);
  return run_ohlc_indicator(natr, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_trange(const float *host_high, const float *host_low,
                     const float *host_close, float *host_output, int size,
                     cudaStream_t stream) {
  tacuda::TRANGE tr;
  return run_ohlc_indicator(tr, host_high, host_low, host_close, host_output,
                            size, 1, stream);
}

ctStatus_t ct_stochastic(const float *host_high, const float *host_low,
                         const float *host_close, float *host_k, float *host_d,
                         int size, int kPeriod, int dPeriod,
                         cudaStream_t stream) {
  tacuda::Stochastic stoch(kPeriod, dPeriod);
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_ohlc_indicator(stoch, host_high, host_low, host_close,
                                     tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_k, tmp.data(), size * sizeof(float));
  std::memcpy(host_d, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_stochf(const float *host_high, const float *host_low,
                     const float *host_close, float *host_k, float *host_d,
                     int size, int kPeriod, int dPeriod,
                     cudaStream_t stream) {
  tacuda::StochasticFast stoch(kPeriod, dPeriod);
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_ohlc_indicator(stoch, host_high, host_low, host_close,
                                     tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_k, tmp.data(), size * sizeof(float));
  std::memcpy(host_d, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_stochrsi(const float *host_input, float *host_k, float *host_d,
                       int size, int rsiPeriod, int kPeriod, int dPeriod,
                       cudaStream_t stream) {
  tacuda::StochRSI st(rsiPeriod, kPeriod, dPeriod);
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(st, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_k, tmp.data(), size * sizeof(float));
  std::memcpy(host_d, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_cci(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::CCI cci(period);
  return run_ohlc_indicator(cci, host_high, host_low, host_close, host_output,
                            size, 1, stream);
}

ctStatus_t ct_adx(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::ADX adx(period);
  return run_ohlc_indicator(adx, host_high, host_low, host_close, host_output,
                            size, 1, stream);
}

ctStatus_t ct_adxr(const float *host_high, const float *host_low,
                   const float *host_close, float *host_output, int size,
                   int period, cudaStream_t stream) {
  tacuda::ADXR adxr(period);
  return run_ohlc_indicator(adxr, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_plus_dm(const float *host_high, const float *host_low,
                      float *host_output, int size, int period, cudaStream_t stream) {
  tacuda::PlusDM pdm(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    pdm.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_minus_dm(const float *host_high, const float *host_low,
                       float *host_output, int size, int period, cudaStream_t stream) {
  tacuda::MinusDM mdm(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mdm.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_plus_di(const float *host_high, const float *host_low,
                      const float *host_close, float *host_output, int size,
                      int period, cudaStream_t stream) {
  tacuda::PlusDI pdi(period);
  return run_ohlc_indicator(pdi, host_high, host_low, host_close, host_output,
                            size, 1, stream);
}

ctStatus_t ct_minus_di(const float *host_high, const float *host_low,
                       const float *host_close, float *host_output, int size,
                       int period, cudaStream_t stream) {
  tacuda::MinusDI mdi(period);
  return run_ohlc_indicator(mdi, host_high, host_low, host_close, host_output,
                            size, 1, stream);
}

ctStatus_t ct_sar(const float *host_high, const float *host_low,
                  float *host_output, int size, float step,
                  float maxAcceleration, cudaStream_t stream) {
  tacuda::SAR sar(step, maxAcceleration);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    sar.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_sarext(const float *host_high, const float *host_low,
                     float *host_output, int size, float startValue,
                     float offsetOnReverse, float accInitLong, float accLong,
                     float accMaxLong, float accInitShort, float accShort,
                     float accMaxShort, cudaStream_t stream) {
  tacuda::SAREXT sar(startValue, offsetOnReverse, accInitLong, accLong, accMaxLong,
             accInitShort, accShort, accMaxShort);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    sar.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_aroon(const float *host_high, const float *host_low,
                    float *host_up, float *host_down, float *host_osc, int size,
                    int upPeriod, int downPeriod, cudaStream_t stream) {
  tacuda::Aroon aroon(upPeriod, downPeriod);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size) * 3)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    aroon.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  std::vector<float> tmpHost(3 * size);
  err = cudaMemcpyAsync(tmpHost.data(), d_out.get(), 3 * size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  std::memcpy(host_up, tmpHost.data(), size * sizeof(float));
  std::memcpy(host_down, tmpHost.data() + size, size * sizeof(float));
  std::memcpy(host_osc, tmpHost.data() + 2 * size, size * sizeof(float));

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_aroonosc(const float *host_high, const float *host_low,
                       float *host_output, int size, int period, cudaStream_t stream) {
  tacuda::AroonOscillator ind(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_adosc(const float *host_high, const float *host_low,
                    const float *host_close, const float *host_volume,
                    float *host_output, int size, int shortPeriod,
                    int longPeriod, cudaStream_t stream) {
  tacuda::ADOSC adosc(shortPeriod, longPeriod);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_vol{nullptr}, d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) ||
      !acquire_buffer(d_vol, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_vol.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    adosc.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                    d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ad(const float *host_high, const float *host_low,
                 const float *host_close, const float *host_volume,
                 float *host_output, int size, cudaStream_t stream) {
  tacuda::AD ad;
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_vol{nullptr}, d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) ||
      !acquire_buffer(d_vol, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_vol.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ad.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                 d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_avgprice(const float *host_open, const float *host_high,
                       const float *host_low, const float *host_close,
                       float *host_output, int size, cudaStream_t stream) {
  tacuda::AvgPrice ap;
  DeviceBuffer d_open{nullptr}, d_high{nullptr}, d_low{nullptr},
      d_close{nullptr}, d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_open, elemCount) || !acquire_buffer(d_high, elemCount) ||
      !acquire_buffer(d_low, elemCount) || !acquire_buffer(d_close, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_open.get(), host_open, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ap.calculate(d_open.get(), d_high.get(), d_low.get(), d_close.get(),
                 d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_medprice(const float *host_high, const float *host_low,
                       float *host_output, int size, cudaStream_t stream) {
  tacuda::MedPrice mp;
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mp.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_typprice(const float *host_high, const float *host_low,
                       const float *host_close, float *host_output, int size, cudaStream_t stream) {
  tacuda::TypPrice tp;
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    tp.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size,
                 stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_wclprice(const float *host_high, const float *host_low,
                       const float *host_close, float *host_output, int size, cudaStream_t stream) {
  tacuda::WclPrice wc;
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    wc.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size,
                 stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_willr(const float *host_high, const float *host_low,
                    const float *host_close, float *host_output, int size,
                    int period, cudaStream_t stream) {
  tacuda::WILLR willr(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    willr.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(),
                    size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_midpoint(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  tacuda::MIDPOINT mp(period);
  return run_indicator(mp, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_midprice(const float *host_high, const float *host_low,
                       float *host_output, int size, int period, cudaStream_t stream) {
  tacuda::MIDPRICE mp(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mp.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ultosc(const float *host_high, const float *host_low,
                     const float *host_close, float *host_output, int size,
                     int shortPeriod, int mediumPeriod, int longPeriod, cudaStream_t stream) {
  tacuda::ULTOSC ultosc(shortPeriod, mediumPeriod, longPeriod);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ultosc.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(),
                     size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_mfi(const float *host_high, const float *host_low,
                  const float *host_close, const float *host_volume,
                  float *host_output, int size, int period, cudaStream_t stream) {
  tacuda::MFI mfi(period);
  DeviceBuffer d_high{nullptr}, d_low{nullptr}, d_close{nullptr},
      d_vol{nullptr}, d_out{nullptr};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_vol, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_vol.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mfi.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                  d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_imi(const float *host_open, const float *host_close,
                  float *host_output, int size, int period, cudaStream_t stream) {
  tacuda::IMI imi(period);
  DeviceBuffer d_open{nullptr}, d_close{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_open, static_cast<size_t>(size)) ||
      !acquire_buffer(d_close, static_cast<size_t>(size)) ||
      !acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_open.get(), host_open, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    imi.calculate(d_open.get(), d_close.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_obv(const float *host_price, const float *host_volume,
                  float *host_output, int size, cudaStream_t stream) {
  tacuda::OBV obv;
  DeviceBuffer d_price{nullptr}, d_volume{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_price, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_volume, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_price.get(), host_price, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_volume.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    obv.calculate(d_price.get(), d_volume.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_nvi(const float *host_close, const float *host_volume,
                  float *host_output, int size, cudaStream_t stream) {
  tacuda::NVI nvi;
  std::vector<float> packed(2 * size);
  std::memcpy(packed.data(), host_close, size * sizeof(float));
  std::memcpy(packed.data() + size, host_volume, size * sizeof(float));
  return run_indicator(nvi, packed.data(), host_output, size, 1, stream);
}

ctStatus_t ct_pvi(const float *host_close, const float *host_volume,
                  float *host_output, int size, cudaStream_t stream) {
  tacuda::PVI pvi;
  std::vector<float> packed(2 * size);
  std::memcpy(packed.data(), host_close, size * sizeof(float));
  std::memcpy(packed.data() + size, host_volume, size * sizeof(float));
  return run_indicator(pvi, packed.data(), host_output, size, 1, stream);
}

ctStatus_t ct_beta(const float *host_x, const float *host_y, float *host_output,
                   int size, int period, cudaStream_t stream) {
  tacuda::Beta beta(period);
  DeviceBuffer d_x{nullptr}, d_y{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_x, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_y, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_x.get(), host_x, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_y.get(), host_y, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    beta.calculate(d_x.get(), d_y.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ht_dcperiod(const float *host_input, float *host_output,
                          int size, cudaStream_t stream) {
  tacuda::HT_DCPERIOD ind;
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_ht_dcphase(const float *host_input, float *host_output,
                         int size, cudaStream_t stream) {
  tacuda::HT_DCPHASE ind;
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_ht_phasor(const float *host_input, float *host_inphase,
                        float *host_quadrature, int size, cudaStream_t stream) {
  tacuda::HT_PHASOR ind;
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(ind, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_inphase, tmp.data(), size * sizeof(float));
  std::memcpy(host_quadrature, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ht_sine(const float *host_input, float *host_sine,
                      float *host_leadsine, int size, cudaStream_t stream) {
  tacuda::HT_SINE ind;
  std::vector<float> tmp(2 * size);
  ctStatus_t rc = run_indicator(ind, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_sine, tmp.data(), size * sizeof(float));
  std::memcpy(host_leadsine, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_ht_trendline(const float *host_input, float *host_output,
                           int size, cudaStream_t stream) {
  tacuda::HT_TRENDLINE ind;
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_ht_trendmode(const float *host_input, float *host_output,
                           int size, cudaStream_t stream) {
  tacuda::HT_TRENDMODE ind;
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_linearreg(const float *host_input, float *host_output, int size,
                        int period, cudaStream_t stream) {
  tacuda::LINEARREG ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_linearreg_slope(const float *host_input, float *host_output,
                              int size, int period, cudaStream_t stream) {
  tacuda::LINEARREG_SLOPE ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_linearreg_intercept(const float *host_input, float *host_output,
                                  int size, int period, cudaStream_t stream) {
  tacuda::LINEARREG_INTERCEPT ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_linearreg_angle(const float *host_input, float *host_output,
                              int size, int period, cudaStream_t stream) {
  tacuda::LINEARREG_ANGLE ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_tsf(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::TSF ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_bop(const float *host_open, const float *host_high,
                  const float *host_low, const float *host_close,
                  float *host_output, int size, cudaStream_t stream) {
  tacuda::BOP bop;
  return run_ohlc_indicator(bop, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_doji(const float *host_open, const float *host_high,
                       const float *host_low, const float *host_close,
                       float *host_output, int size, cudaStream_t stream) {
  tacuda::Doji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_hammer(const float *host_open, const float *host_high,
                         const float *host_low, const float *host_close,
                         float *host_output, int size, cudaStream_t stream) {
  tacuda::Hammer ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_inverted_hammer(const float *host_open,
                                  const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream) {
  tacuda::InvertedHammer ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_bullish_engulfing(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  tacuda::BullishEngulfing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_bearish_engulfing(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  tacuda::BearishEngulfing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_three_white_soldiers(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream) {
  tacuda::ThreeWhiteSoldiers ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_abandoned_baby(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  tacuda::AbandonedBaby ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_advance_block(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  tacuda::AdvanceBlock ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_belt_hold(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::BeltHold ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_breakaway(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::Breakaway ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_two_crows(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::TwoCrows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_three_black_crows(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  tacuda::ThreeBlackCrows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_three_inside(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  tacuda::ThreeInside ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_three_outside(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  tacuda::CDL3OUTSIDE ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_three_line_strike(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  tacuda::ThreeLineStrike ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_three_stars_in_south(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream) {
  tacuda::ThreeStarsInSouth ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_closing_marubozu(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  tacuda::ClosingMarubozu ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_conceal_baby_swallow(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream) {
  tacuda::ConcealBabySwallow ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_counterattack(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  tacuda::CounterAttack ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_dark_cloud_cover(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  tacuda::DarkCloudCover ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_doji_star(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::DojiStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_dragonfly_doji(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  tacuda::DragonflyDoji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_engulfing(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::Engulfing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_evening_doji_star(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  tacuda::EveningDojiStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_evening_star(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  tacuda::EveningStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_gap_side_side_white(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size, cudaStream_t stream) {
  tacuda::GapSideSideWhite ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_gravestone_doji(const float *host_open,
                                  const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream) {
  tacuda::GravestoneDoji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_hanging_man(const float *host_open, const float *host_high,
                              const float *host_low, const float *host_close,
                              float *host_output, int size, cudaStream_t stream) {
  tacuda::HangingMan ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_harami(const float *host_open, const float *host_high,
                         const float *host_low, const float *host_close,
                         float *host_output, int size, cudaStream_t stream) {
  tacuda::Harami ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_harami_cross(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  tacuda::HaramiCross ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_high_wave(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::HighWave ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_hikkake(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  tacuda::Hikkake ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_hikkake_mod(const float *host_open, const float *host_high,
                              const float *host_low, const float *host_close,
                              float *host_output, int size, cudaStream_t stream) {
  tacuda::HikkakeMod ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_homing_pigeon(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  tacuda::HomingPigeon ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_identical_three_crows(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size, cudaStream_t stream) {
  tacuda::IdenticalThreeCrows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_in_neck(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  tacuda::InNeck ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_kicking(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  tacuda::Kicking ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_kicking_by_length(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  tacuda::KickingByLength ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_ladder_bottom(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  tacuda::LadderBottom ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_long_legged_doji(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  tacuda::LongLeggedDoji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_long_line(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::LongLine ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_marubozu(const float *host_open, const float *host_high,
                           const float *host_low, const float *host_close,
                           float *host_output, int size, cudaStream_t stream) {
  tacuda::Marubozu ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_matching_low(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  tacuda::MatchingLow ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_mat_hold(const float *host_open, const float *host_high,
                           const float *host_low, const float *host_close,
                           float *host_output, int size, cudaStream_t stream) {
  tacuda::MatHold ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_morning_doji_star(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  tacuda::MorningDojiStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_morning_star(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  tacuda::MorningStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_on_neck(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  tacuda::OnNeck ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_piercing(const float *host_open, const float *host_high,
                           const float *host_low, const float *host_close,
                           float *host_output, int size, cudaStream_t stream) {
  tacuda::Piercing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_rickshaw_man(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  tacuda::RickshawMan ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_rise_fall3_methods(const float *host_open,
                                     const float *host_high,
                                     const float *host_low,
                                     const float *host_close,
                                     float *host_output, int size, cudaStream_t stream) {
  tacuda::RiseFall3Methods ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_separating_lines(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  tacuda::SeparatingLines ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_shooting_star(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  tacuda::ShootingStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_short_line(const float *host_open, const float *host_high,
                             const float *host_low, const float *host_close,
                             float *host_output, int size, cudaStream_t stream) {
  tacuda::ShortLine ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_spinning_top(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  tacuda::SpinningTop ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_stalled_pattern(const float *host_open,
                                  const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream) {
  tacuda::StalledPattern ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_stick_sandwich(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  tacuda::StickSandwich ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_takuri(const float *host_open, const float *host_high,
                         const float *host_low, const float *host_close,
                         float *host_output, int size, cudaStream_t stream) {
  tacuda::Takuri ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_tasuki_gap(const float *host_open, const float *host_high,
                             const float *host_low, const float *host_close,
                             float *host_output, int size, cudaStream_t stream) {
  tacuda::TasukiGap ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_thrusting(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  tacuda::Thrusting ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_tristar(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  tacuda::Tristar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_unique_3_river(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  tacuda::Unique3River ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_upside_gap_2_crows(const float *host_open,
                                     const float *host_high,
                                     const float *host_low,
                                     const float *host_close,
                                     float *host_output, int size, cudaStream_t stream) {
  tacuda::UpsideGap2Crows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cdl_xside_gap_3_methods(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size, cudaStream_t stream) {
  tacuda::XSideGap3Methods ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);
}

ctStatus_t ct_cmo(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  tacuda::CMO cmo(period);
  return run_indicator(cmo, host_input, host_output, size, 1, stream);
}

ctStatus_t ct_correl(const float *host_x, const float *host_y,
                     float *host_output, int size, int period, cudaStream_t stream) {
  tacuda::Correl correl(period);
  DeviceBuffer d_x{nullptr}, d_y{nullptr}, d_out{nullptr};

  if (!acquire_buffer(d_x, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_y, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_x.get(), host_x, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_y.get(), host_y, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    correl.calculate(d_x.get(), d_y.get(), d_out.get(), size, stream);
  } catch (...) {
    return CT_STATUS_KERNEL_FAILED;
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;
}

ctStatus_t ct_dx(const float *host_high, const float *host_low,
                 const float *host_close, float *host_output, int size,
                 int period, cudaStream_t stream) {
  tacuda::DX dx(period);
  return run_ohlc_indicator(dx, host_high, host_low, host_close, host_output,
                            size, 1, stream);
}

} // extern "C"
