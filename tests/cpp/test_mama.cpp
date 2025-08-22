#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, MAMA) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> mama(N, 0.0f), fama(N, 0.0f);
  float fast = 0.5f, slow = 0.05f;
  ctStatus_t rc = ct_mama(x.data(), mama.data(), fama.data(), N, fast, slow);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  std::vector<float> mamaRef(N), famaRef(N);
  mamaRef[0] = x[0];
  famaRef[0] = x[0];
  for (int i = 1; i < N; ++i) {
    mamaRef[i] = fast * x[i] + (1.0f - fast) * mamaRef[i - 1];
    famaRef[i] = slow * mamaRef[i] + (1.0f - slow) * famaRef[i - 1];
  }
  expect_approx_equal(mama, mamaRef);
  expect_approx_equal(fama, famaRef);
}
