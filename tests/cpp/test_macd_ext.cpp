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
