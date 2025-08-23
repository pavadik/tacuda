#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLTasukiGap) {
  const int N = 4;
  std::vector<float> open{10.0f, 11.5f, 11.7f, 12.0f};
  std::vector<float> high{11.2f, 12.2f, 11.8f, 12.5f};
  std::vector<float> low{9.5f, 11.5f, 11.2f, 11.8f};
  std::vector<float> close{11.0f, 12.0f, 11.3f, 11.7f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_tasuki_gap(open.data(), high.data(), low.data(),
                                    close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_tasuki_gap failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

