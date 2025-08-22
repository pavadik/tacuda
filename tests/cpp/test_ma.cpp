#include <tacuda.h>
#include "test_utils.hpp"

static std::vector<float> ema_ref(const std::vector<float>& in, int period) {
  std::vector<float> out(in.size(), std::numeric_limits<float>::quiet_NaN());
  const float k = 2.0f / (period + 1.0f);
  for (size_t i = 0; i + period <= in.size(); ++i) {
    float weight = 1.0f;
    float weightedSum = in[i + period - 1];
    float weightSum = 1.0f;
    for (int j = 1; j < period; ++j) {
      weight *= (1.0f - k);
      weightedSum += in[i + period - 1 - j] * weight;
      weightSum += weight;
    }
    out[i] = weightedSum / weightSum;
  }
  return out;
}

TEST(Tacuda, MA_SMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_ma(x.data(), out.data(), N, p, CT_MA_SMA);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float s = 0.0f;
    for (int k = 0; k < p; ++k)
      s += x[i + k];
    ref[i] = s / p;
  }
  expect_approx_equal(out, ref);
}

TEST(Tacuda, MA_EMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> out(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_ma(x.data(), out.data(), N, p, CT_MA_EMA);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  auto ref = ema_ref(x, p);
  expect_approx_equal(out, ref);
}
