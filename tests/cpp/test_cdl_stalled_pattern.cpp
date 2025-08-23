#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLStalledPattern) {
  const int N = 4;
  std::vector<float> open{1.0f, 2.1f, 3.3f, 4.0f};
  std::vector<float> high{2.2f, 3.5f, 3.6f, 4.6f};
  std::vector<float> low{0.8f, 2.0f, 3.2f, 3.8f};
  std::vector<float> close{2.0f, 3.2f, 3.4f, 4.5f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_stalled_pattern(open.data(), high.data(), low.data(),
                                         close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_stalled_pattern failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

