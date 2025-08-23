#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLOnNeck) {
  const int N = 2;
  std::vector<float> open{10.0f, 9.4f};
  std::vector<float> high{10.5f, 9.8f};
  std::vector<float> low{9.5f, 9.3f};
  std::vector<float> close{9.7f, 9.5f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_on_neck(open.data(), high.data(), low.data(),
                                 close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_on_neck failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0])) << "expected NaN at head 0";
}
