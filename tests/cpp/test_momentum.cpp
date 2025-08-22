#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, Momentum) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_momentum(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_momentum failed";
  for (int i = 0; i < N - p; i++)
    ref[i] = x[i + p] - x[i];
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}
