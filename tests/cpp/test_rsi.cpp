#include <tacuda.h>
#include "test_utils.hpp"

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
