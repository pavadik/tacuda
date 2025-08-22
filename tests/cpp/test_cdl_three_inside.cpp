#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThreeInside) {
  const int N = 5;
  std::vector<float> open{10.0f, 11.5f, 11.3f, 12.4f, 12.3f};
  std::vector<float> high{12.2f, 11.6f, 12.6f, 12.5f, 12.4f};
  std::vector<float> low{9.8f, 11.1f, 11.0f, 11.9f, 11.8f};
  std::vector<float> close{12.0f, 11.2f, 12.5f, 12.3f, 12.2f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_three_inside(open.data(), high.data(), low.data(),
                                      close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_three_inside failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

