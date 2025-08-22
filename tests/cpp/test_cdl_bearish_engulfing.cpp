#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLBearishEngulfing) {
  const int N = 4;
  std::vector<float> open{10.0f, 10.5f, 10.9f, 10.8f};
  std::vector<float> high{10.4f, 11.0f, 11.1f, 11.0f};
  std::vector<float> low{9.8f, 10.4f, 10.0f, 10.5f};
  std::vector<float> close{10.2f, 10.8f, 10.2f, 10.6f};
  std::vector<float> out(N);
  std::vector<float> ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_bearish_engulfing(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_bearish_engulfing failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
}
