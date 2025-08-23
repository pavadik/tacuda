#include "test_utils.hpp"
#include <cmath>
#include <tacuda.h>

TEST(Tacuda, MININDEX) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_minindex(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float minVal = x[i];
    int minIdx = 0;
    for (int k = 1; k < p; ++k) {
      if (x[i + k] < minVal) {
        minVal = x[i + k];
        minIdx = k;
      }
    }
    ref[i] = static_cast<float>(minIdx);
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i]));
}
