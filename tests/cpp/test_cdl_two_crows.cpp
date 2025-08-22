#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLTwoCrows) {
  const int N = 5;
  std::vector<float> open{10.0f, 12.0f, 11.8f, 11.5f, 11.5f};
  std::vector<float> high{11.2f, 12.3f, 12.0f, 11.6f, 11.6f};
  std::vector<float> low{9.8f, 11.4f, 10.5f, 11.4f, 11.4f};
  std::vector<float> close{11.0f, 11.6f, 10.6f, 11.5f, 11.5f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_two_crows(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_two_crows failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

