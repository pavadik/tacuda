#include <tacuda.h>
#include "test_utils.hpp"

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
