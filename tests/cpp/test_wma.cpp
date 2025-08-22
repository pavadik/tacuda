#include <tacuda.h>
#include "test_utils.hpp"

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
