#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLAbandonedBaby) {
  const int N = 5;
  std::vector<float> open{10.0f, 8.5f, 9.2f, 9.0f, 9.0f};
  std::vector<float> high{10.1f, 8.6f, 9.9f, 9.1f, 9.1f};
  std::vector<float> low{8.9f, 8.4f, 9.1f, 8.9f, 8.9f};
  std::vector<float> close{9.0f, 8.5f, 9.8f, 9.0f, 9.0f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_abandoned_baby(open.data(), high.data(), low.data(),
                                        close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_abandoned_baby failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
  EXPECT_TRUE(std::isnan(out[1])) << "expected NaN at head 1";
}

