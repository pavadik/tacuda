#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLIdenticalThreeCrows) {
  const int N = 5;
  std::vector<float> open{10.0f, 9.5f, 9.0f, 8.0f, 8.0f};
  std::vector<float> high{10.1f, 9.6f, 9.1f, 8.1f, 8.1f};
  std::vector<float> low{8.9f, 8.4f, 8.4f, 7.9f, 7.9f};
  std::vector<float> close{9.0f, 8.5f, 8.5f, 8.0f, 8.0f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_identical_three_crows(
      open.data(), high.data(), low.data(), close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_identical_three_crows failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}
