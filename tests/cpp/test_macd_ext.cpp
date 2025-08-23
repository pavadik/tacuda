#include <tacuda.h>
#include "test_utils.hpp"

static float ema_at(const std::vector<float>& x, int idx, int period, int start = 0) {
  const float k = 2.0f / (period + 1.0f);
  float weight = 1.0f;
  float weightedSum = x[idx];
  float weightSum = 1.0f;
  int steps = std::min(period, idx - start);
  for (int i = 1; i <= steps; ++i) {
    weight *= (1.0f - k);
    weightedSum += x[idx - i] * weight;
    weightSum += weight;
  }
  return weightedSum / weightSum;
}

TEST(Tacuda, MACD_Ext) {
  const int N = 64;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> macd(N, 0.0f), signal(N, 0.0f), hist(N, 0.0f);
  int fastP = 3, slowP = 8, signalP = 4;
  ctStatus_t rc = ct_macd(x.data(), macd.data(), signal.data(), hist.data(), N,
                          fastP, slowP, signalP, CT_MA_EMA);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  std::vector<float> macdRef(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = slowP; i < N; ++i) {
    float fast = ema_at(x, i, fastP, 0);
    float slow = ema_at(x, i, slowP, 0);
    macdRef[i] = fast - slow;
  }
  std::vector<float> sigRef(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> histRef(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = slowP; i < N; ++i) {
    float sig = ema_at(macdRef, i, signalP, slowP);
    sigRef[i] = sig;
    histRef[i] = macdRef[i] - sig;
  }
  expect_approx_equal(macd, macdRef);
  expect_approx_equal(signal, sigRef);
  expect_approx_equal(hist, histRef);
}

TEST(Tacuda, MACD_ExtLargePeriod) {
  const int N = 200000;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.001f * i);

  int fastP = 5000, slowP = 10000, signalP = 4000;
  std::vector<float> macd(N, 0.0f), signal(N, 0.0f), hist(N, 0.0f);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  ctStatus_t rc = ct_macd(x.data(), macd.data(), signal.data(), hist.data(), N,
                          fastP, slowP, signalP, CT_MA_EMA);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  EXPECT_LT(ms, 400.0f);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);

  std::vector<float> emaFast(N), emaSlow(N), macdRef(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> sigRef(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> histRef(N, std::numeric_limits<float>::quiet_NaN());
  float kFast = 2.0f / (fastP + 1.0f);
  float kSlow = 2.0f / (slowP + 1.0f);
  float kSig = 2.0f / (signalP + 1.0f);
  emaFast[0] = emaSlow[0] = x[0];
  for (int i = 1; i < N; ++i) {
    emaFast[i] = kFast * x[i] + (1.0f - kFast) * emaFast[i - 1];
    emaSlow[i] = kSlow * x[i] + (1.0f - kSlow) * emaSlow[i - 1];
    if (i >= slowP) {
      macdRef[i] = emaFast[i] - emaSlow[i];
      if (i == slowP)
        sigRef[i] = macdRef[i];
      else
        sigRef[i] = kSig * macdRef[i] + (1.0f - kSig) * sigRef[i - 1];
      histRef[i] = macdRef[i] - sigRef[i];
    }
  }
  expect_approx_equal(macd, macdRef, 1e-3f);
  expect_approx_equal(signal, sigRef, 1e-3f);
  expect_approx_equal(hist, histRef, 1e-3f);
  for (int i = 0; i < slowP; ++i) {
    EXPECT_TRUE(std::isnan(macd[i]));
    EXPECT_TRUE(std::isnan(signal[i]));
    EXPECT_TRUE(std::isnan(hist[i]));
  }
}
