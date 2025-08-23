#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLKickingByLength) {
  const int N = 3;
  std::vector<float> open{10.0f, 11.0f, 11.0f};
  std::vector<float> high{10.0f, 12.0f, 12.0f};
  std::vector<float> low{9.5f, 11.0f, 11.0f};
  std::vector<float> close{9.5f, 12.0f, 11.5f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_kicking_by_length(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_kicking_by_length failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
}
