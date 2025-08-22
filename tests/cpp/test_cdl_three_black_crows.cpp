#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThreeBlackCrows) {
  const int N = 5;
  std::vector<float> open{10.0f, 9.5f, 9.0f, 7.6f, 7.6f};
  std::vector<float> high{10.1f, 9.6f, 9.1f, 7.7f, 7.7f};
  std::vector<float> low{8.9f, 8.4f, 7.4f, 7.5f, 7.5f};
  std::vector<float> close{9.0f, 8.5f, 7.5f, 7.6f, 7.6f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_three_black_crows(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_three_black_crows failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

