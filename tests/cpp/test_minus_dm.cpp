#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, MINUS_DM) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int p = 3;
  ctStatus_t rc = ct_minus_dm(high.data(), low.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_minus_dm failed";
  auto ref = minus_dm_ref(high, low, p);
  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
