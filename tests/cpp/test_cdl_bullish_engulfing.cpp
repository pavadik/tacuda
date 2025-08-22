#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLBullishEngulfing) {
  const int N = 4;
  std::vector<float> open{10.0f, 9.5f, 9.1f, 9.3f};
  std::vector<float> high{10.2f, 9.7f, 9.9f, 9.5f};
  std::vector<float> low{9.7f, 9.1f, 9.0f, 8.9f};
  std::vector<float> close{9.8f, 9.2f, 9.8f, 9.1f};
  std::vector<float> out(N);
  std::vector<float> ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_bullish_engulfing(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_bullish_engulfing failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
}
