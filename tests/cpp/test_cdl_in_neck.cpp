#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLInNeck) {
  const int N = 3;
  std::vector<float> open{10.0f, 8.8f, 9.0f};
  std::vector<float> high{10.1f, 9.1f, 9.2f};
  std::vector<float> low{8.9f, 8.7f, 8.8f};
  std::vector<float> close{9.0f, 9.05f, 9.1f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_in_neck(open.data(), high.data(), low.data(),
                                 close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_in_neck failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
}
