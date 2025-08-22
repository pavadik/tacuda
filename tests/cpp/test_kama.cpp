#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, KAMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 10;
  int fastP = 2;
  int slowP = 30;
  ctStatus_t rc = ct_kama(x.data(), out.data(), N, p, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_kama failed";

  auto ref = kama_ref(x, p, fastP, slowP);
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}
