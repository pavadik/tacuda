#include <algorithm>
#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <tacuda.h>
#include <vector>

namespace {

// Helper to compare floating point vectors, ignoring NaNs.
void expect_approx_equal(const std::vector<float> &a,
                         const std::vector<float> &b, float eps = 1e-3f) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i]))
      continue;
    EXPECT_NEAR(a[i], b[i], eps) << "Mismatch at index " << i;
  }
}

std::vector<float> sar_ref(const std::vector<float> &high,
                           const std::vector<float> &low,
                           float step, float maxAcc) {
  size_t n = high.size();
  std::vector<float> out(n);
  float af = step;
  float ep = high[0];
  float sar = low[0];
  bool longPos = true;
  out[0] = sar;
  for (size_t i = 1; i < n; ++i) {
    sar = sar + af * (ep - sar);
    if (longPos) {
      sar = std::min(sar, low[i - 1]);
      if (low[i] < sar) {
        longPos = false;
        sar = ep;
        ep = low[i];
        af = step;
        sar = std::max(sar, high[i - 1]);
      } else {
        if (high[i] > ep) {
          ep = high[i];
          af = std::min(af + step, maxAcc);
        }
      }
    } else {
      sar = std::max(sar, high[i - 1]);
      if (high[i] > sar) {
        longPos = true;
        sar = ep;
        ep = high[i];
        af = step;
        sar = std::min(sar, low[i - 1]);
      } else {
        if (low[i] < ep) {
          ep = low[i];
          af = std::min(af + step, maxAcc);
        }
      }
    }
    out[i] = sar;
  }
  return out;
}

} // namespace

TEST(Tacuda, SMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_sma(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sma failed";
  for (int i = 0; i <= N - p; i++) {
    float s = 0.0f;
    for (int k = 0; k < p; k++)
      s += x[i + k];
    ref[i] = s / p;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, WMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_wma(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_wma failed";
  float denom = 0.5f * p * (p + 1);
  for (int i = 0; i <= N - p; i++) {
    float s = 0.0f;
    for (int k = 0; k < p; k++)
      s += x[i + k] * (p - k);
    ref[i] = s / denom;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, EMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_ema(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ema failed";

  const float k = 2.0f / (p + 1.0f);
  for (int i = 0; i <= N - p; ++i) {
    float weight = 1.0f;
    float weightedSum = x[i + p - 1];
    float weightSum = 1.0f;
    for (int j = 1; j < p; ++j) {
      weight *= (1.0f - k);
      weightedSum += x[i + p - 1 - j] * weight;
      weightSum += weight;
    }
    ref[i] = weightedSum / weightSum;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, Momentum) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_momentum(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_momentum failed";
  for (int i = 0; i < N - p; i++)
    ref[i] = x[i + p] - x[i];
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, MacdLine) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int fastP = 12, slowP = 26;
  ctStatus_t rc = ct_macd_line(x.data(), out.data(), N, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_macd_line failed";
  for (int i = 0; i < slowP; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
  for (int i = slowP; i < N; i++) {
    EXPECT_TRUE(std::isfinite(out[i])) << "expected finite value at " << i;
  }
}

TEST(Tacuda, RSI) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 14;
  ctStatus_t rc = ct_rsi(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_rsi failed";
  for (int i = 0; i < N - p; ++i) {
    float gain = 0.0f;
    float loss = 0.0f;
    for (int j = 0; j < p; ++j) {
      float diff = x[i + j + 1] - x[i + j];
      if (diff > 0.0f)
        gain += diff;
      else
        loss -= diff;
    }
    float avgGain = gain / p;
    float avgLoss = loss / p;
    float rsi;
    if (avgLoss == 0.0f)
      rsi = (avgGain == 0.0f) ? 50.0f : 100.0f;
    else if (avgGain == 0.0f)
      rsi = 0.0f;
    else {
      float rs = avgGain / avgLoss;
      rsi = 100.0f - 100.0f / (1.0f + rs);
    }
    ref[i] = rsi;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, BBANDS) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> upper(N, 0.0f), middle(N, 0.0f), lower(N, 0.0f);
  std::vector<float> refU(N, 0.0f), refM(N, 0.0f), refL(N, 0.0f);

  int p = 20;
  float up = 2.0f, down = 2.0f;
  ctStatus_t rc = ct_bbands(x.data(), upper.data(), middle.data(), lower.data(),
                            N, p, up, down);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_bbands failed";

  for (int i = 0; i <= N - p; ++i) {
    float sum = 0.0f, sumSq = 0.0f;
    for (int j = 0; j < p; ++j) {
      float v = x[i + j];
      sum += v;
      sumSq += v * v;
    }
    float mean = sum / p;
    float var = sumSq / p - mean * mean;
    var = std::max(var, 0.0f);
    float stddev = std::sqrt(var);
    refM[i] = mean;
    refU[i] = mean + up * stddev;
    refL[i] = mean - down * stddev;
  }

  expect_approx_equal(upper, refU);
  expect_approx_equal(middle, refM);
  expect_approx_equal(lower, refL);

  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(upper[i])) << "expected NaN at tail " << i;
    EXPECT_TRUE(std::isnan(middle[i])) << "expected NaN at tail " << i;
    EXPECT_TRUE(std::isnan(lower[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, ATR) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low  = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                             48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                             49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close= {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                             49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                             49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f), ref(N,
                                     std::numeric_limits<float>::quiet_NaN());

  int p = 14;
  ctStatus_t rc = ct_atr(high.data(), low.data(), close.data(), out.data(), N, p, 0.0f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_atr failed";

  std::vector<float> tr(N);
  tr[0] = high[0] - low[0];
  for (int i = 1; i < N; ++i) {
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float sum = 0.0f;
  for (int i = 0; i < p; ++i)
    sum += tr[i];
  float atr = sum / p;
  ref[p - 1] = atr;
  for (int i = p; i < N; ++i) {
    atr = (atr * (p - 1) + tr[i]) / p;
    ref[i] = atr;
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, Stochastic) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low  = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                             48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                             49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close= {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                             49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                             49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> k(N, 0.0f), d(N, 0.0f);

  int kP = 5, dP = 3;
  ctStatus_t rc = ct_stochastic(high.data(), low.data(), close.data(),
                                k.data(), d.data(), N, kP, dP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_stochastic failed";

  std::vector<float> refK = {
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
      86.45833f, 97.29730f, 99.40476f, 81.93548f,
      40.60150f, 48.12030f, 65.89147f, 75.19380f, 84.24658f};
  std::vector<float> refD = {
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
      89.94871f, 93.85261f, 94.38680f, 92.87918f,
      73.98058f, 56.88576f, 51.53776f, 63.06852f, 75.11062f};

  expect_approx_equal(k, refK);
  expect_approx_equal(d, refD);
  for (int i = 0; i < kP + dP - 2; ++i) {
    EXPECT_TRUE(std::isnan(k[i])) << "expected NaN at head " << i;
    EXPECT_TRUE(std::isnan(d[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, CCI) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low  = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                             48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                             49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close= {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                             49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                             49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc = ct_cci(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cci failed";

  for (int i = p - 1; i < N; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < p; ++j) {
      int idx = i - j;
      sum += (high[idx] + low[idx] + close[idx]) / 3.0f;
    }
    float sma = sum / p;
    float dev = 0.0f;
    for (int j = 0; j < p; ++j) {
      int idx = i - j;
      float tp = (high[idx] + low[idx] + close[idx]) / 3.0f;
      dev += std::fabs(tp - sma);
    }
    float md = dev / p;
    float tp_cur = (high[i] + low[i] + close[i]) / 3.0f;
    ref[i] = (md == 0.0f) ? 0.0f : (tp_cur - sma) / (0.015f * md);
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, OBV) {
  std::vector<float> price = {1.0f, 2.0f, 2.0f, 1.0f, 3.0f};
  std::vector<float> volume = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  const int N = price.size();
  std::vector<float> out(N, 0.0f), ref = {10.0f, 30.0f, 30.0f, -10.0f, 40.0f};

  ctStatus_t rc = ct_obv(price.data(), volume.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_obv failed";
  expect_approx_equal(out, ref);
}

namespace {
std::vector<float> adosc_ref(const std::vector<float>& high,
                             const std::vector<float>& low,
                             const std::vector<float>& close,
                             const std::vector<float>& volume,
                             int shortP, int longP) {
  int N = high.size();
  std::vector<float> ad(N, 0.0f);
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  float cum = 0.0f;
  for (int i = 0; i < N; ++i) {
    float denom = high[i] - low[i];
    float clv = denom == 0.0f ? 0.0f : ((close[i] - low[i]) - (high[i] - close[i])) / denom;
    cum += clv * volume[i];
    ad[i] = cum;
  }
  auto ema_at = [](const std::vector<float>& x, int idx, int period) {
    const float k = 2.0f / (period + 1.0f);
    float weight = 1.0f;
    float weightedSum = x[idx];
    float weightSum = 1.0f;
    int steps = std::min(period, idx);
    for (int i = 1; i <= steps; ++i) {
      weight *= (1.0f - k);
      weightedSum += x[idx - i] * weight;
      weightSum += weight;
    }
    return weightedSum / weightSum;
  };
  for (int i = longP; i < N; ++i) {
    float emaS = ema_at(ad, i, shortP);
    float emaL = ema_at(ad, i, longP);
    out[i] = emaS - emaL;
  }
  return out;
}
} // namespace

TEST(Tacuda, ADOSC) {
  std::vector<float> high   = {12.f, 12.5f, 13.f, 13.5f, 14.f, 14.5f};
  std::vector<float> low    = {11.f, 11.5f, 12.f, 12.5f, 13.f, 13.5f};
  std::vector<float> close  = {11.5f, 12.f, 12.5f, 13.f, 13.5f, 14.f};
  std::vector<float> volume = {100.f, 110.f, 120.f, 130.f, 140.f, 150.f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int shortP = 3, longP = 5;
  ctStatus_t rc = ct_adosc(high.data(), low.data(), close.data(), volume.data(),
                           out.data(), N, shortP, longP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_adosc failed";
  auto ref = adosc_ref(high, low, close, volume, shortP, longP);
  expect_approx_equal(out, ref);
  for (int i = 0; i < longP; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, ADX) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low  = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                             34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close= {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                             35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f), ref(N,
                                     std::numeric_limits<float>::quiet_NaN());

  int p = 3;
  ctStatus_t rc = ct_adx(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_adx failed";

  std::vector<float> dmp(N, 0.0f), dmm(N, 0.0f), tr(N, 0.0f), dx(N, 0.0f);
  for (int i = 1; i < N; ++i) {
    float upMove = high[i] - high[i - 1];
    float downMove = low[i - 1] - low[i];
    dmp[i] = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
    dmm[i] = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float dmp_s = 0.0f, dmm_s = 0.0f, tr_s = 0.0f;
  for (int i = 1; i <= p; ++i) {
    dmp_s += dmp[i];
    dmm_s += dmm[i];
    tr_s += tr[i];
  }
  float dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
  float dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
  dx[p] = (dip + dim == 0.0f) ? 0.0f : 100.0f * std::fabs(dip - dim) / (dip + dim);
  float dx_sum = dx[p];
  for (int i = p + 1; i < N; ++i) {
    dmp_s = dmp_s - dmp_s / p + dmp[i];
    dmm_s = dmm_s - dmm_s / p + dmm[i];
    tr_s = tr_s - tr_s / p + tr[i];
    dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
    dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
    dx[i] = (dip + dim == 0.0f) ? 0.0f :
            100.0f * std::fabs(dip - dim) / (dip + dim);
    if (i < 2 * p) {
      dx_sum += dx[i];
      if (i == 2 * p - 1)
        ref[i] = dx_sum / p;
    } else {
      ref[i] = ((ref[i - 1] * (p - 1)) + dx[i]) / p;
    }
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < 2 * p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, SARTrending) {
  std::vector<float> high = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f};
  std::vector<float> low  = {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sar(high.data(), low.data(), out.data(), N, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sar failed";
  auto ref = sar_ref(high, low, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, SARRanging) {
  std::vector<float> high = {5.f,6.f,5.5f,6.2f,5.8f,6.4f,5.9f,6.5f};
  std::vector<float> low  = {4.f,5.f,4.5f,5.2f,4.8f,5.4f,4.9f,5.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sar(high.data(), low.data(), out.data(), N, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sar failed";
  auto ref = sar_ref(high, low, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, Aroon) {
  std::vector<float> high = {1.f,2.f,3.f,2.f,3.f,4.f,5.f,4.f,6.f,7.f};
  std::vector<float> low  = {0.5f,1.5f,2.5f,1.5f,2.5f,3.5f,4.5f,3.5f,5.5f,6.5f};
  const int N = high.size();
  std::vector<float> up(N, 0.0f), down(N, 0.0f), osc(N, 0.0f);
  int pUp = 5, pDown = 5;
  ctStatus_t rc = ct_aroon(high.data(), low.data(), up.data(), down.data(), osc.data(), N, pUp, pDown);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_aroon failed";

  std::vector<float> refUp(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refDown(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = pUp; i < N; ++i) {
    int sinceHigh = 0; float maxVal = high[i];
    for (int j = 1; j <= pUp; ++j) {
      float val = high[i - j];
      if (val >= maxVal) { maxVal = val; sinceHigh = j; }
    }
    refUp[i] = 100.0f * (pUp - sinceHigh) / pUp;
  }
  for (int i = pDown; i < N; ++i) {
    int sinceLow = 0; float minVal = low[i];
    for (int j = 1; j <= pDown; ++j) {
      float val = low[i - j];
      if (val <= minVal) { minVal = val; sinceLow = j; }
    }
    refDown[i] = 100.0f * (pDown - sinceLow) / pDown;
  }
  std::vector<float> refOsc(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = std::max(pUp, pDown); i < N; ++i)
    refOsc[i] = refUp[i] - refDown[i];

  expect_approx_equal(up, refUp);
  expect_approx_equal(down, refDown);
  expect_approx_equal(osc, refOsc);

  for (int i = 0; i < pUp; ++i)
    EXPECT_TRUE(std::isnan(up[i])) << "expected NaN at head " << i;
  for (int i = 0; i < pDown; ++i)
    EXPECT_TRUE(std::isnan(down[i])) << "expected NaN at head " << i;
  for (int i = 0; i < std::max(pUp, pDown); ++i)
    EXPECT_TRUE(std::isnan(osc[i])) << "expected NaN at head " << i;
}
