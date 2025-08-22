#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, SUM) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_sum(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sum failed";
  for (int i = 0; i <= N - p; ++i) {
    float s = 0.0f;
    for (int k = 0; k < p; ++k)
      s += x[i + k];
    ref[i] = s;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}
