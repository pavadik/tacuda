#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, TRIMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f),
      tmp(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc = ct_trima(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_trima failed";
  int p1 = (p + 1) / 2;
  int p2 = (p % 2 == 0) ? (p1 + 1) : p1;
  for (int i = 0; i <= N - p1; ++i) {
    float s = 0.0f;
    for (int j = 0; j < p1; ++j)
      s += x[i + j];
    tmp[i] = s / p1;
  }
  for (int i = 0; i <= N - p; ++i) {
    float s = 0.0f;
    for (int j = 0; j < p2; ++j)
      s += tmp[i + j];
    ref[i] = s / p2;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}
