#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThrusting) {
  const int N = 3;
  std::vector<float> open{10.0f, 8.8f, 9.5f};
  std::vector<float> high{10.5f, 9.6f, 9.7f};
  std::vector<float> low{9.0f, 8.5f, 9.3f};
  std::vector<float> close{9.2f, 9.4f, 9.6f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_thrusting(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_thrusting failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
}

