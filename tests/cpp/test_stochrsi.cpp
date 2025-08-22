#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, StochRSI) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> k(N, 0.0f), d(N, 0.0f);
  std::vector<float> refK(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refD(N, std::numeric_limits<float>::quiet_NaN());

  int rsiP = 14, kP = 5, dP = 3;
  ctStatus_t rc = ct_stochrsi(x.data(), k.data(), d.data(), N, rsiP, kP, dP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_stochrsi failed";

  std::vector<float> rsi(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = rsiP; i < N; ++i) {
    float gain = 0.0f, loss = 0.0f;
    for (int j = 0; j < rsiP; ++j) {
      float diff = x[i - j] - x[i - j - 1];
      if (diff > 0.0f)
        gain += diff;
      else
        loss -= diff;
    }
    float avgGain = gain / rsiP;
    float avgLoss = loss / rsiP;
    float val;
    if (avgLoss == 0.0f)
      val = (avgGain == 0.0f) ? 50.0f : 100.0f;
    else if (avgGain == 0.0f)
      val = 0.0f;
    else {
      float rs = avgGain / avgLoss;
      val = 100.0f - 100.0f / (1.0f + rs);
    }
    rsi[i] = val;
    if (i >= rsiP + kP - 1) {
      float highest = rsi[i];
      float lowest = rsi[i];
      for (int j = 1; j < kP; ++j) {
        float v = rsi[i - j];
        if (v > highest)
          highest = v;
        if (v < lowest)
          lowest = v;
      }
      float denom = highest - lowest;
      refK[i] = denom == 0.0f ? 0.0f : (rsi[i] - lowest) / denom * 100.0f;
      if (i >= rsiP + kP + dP - 2) {
        float sum = 0.0f;
        for (int j = 0; j < dP; ++j)
          sum += refK[i - j];
        refD[i] = sum / dP;
      }
    }
  }
  expect_approx_equal(k, refK);
  expect_approx_equal(d, refD);
  for (int i = 0; i < rsiP + kP + dP - 2; ++i) {
    EXPECT_TRUE(std::isnan(k[i])) << "expected NaN at head " << i;
    EXPECT_TRUE(std::isnan(d[i])) << "expected NaN at head " << i;
  }
}
