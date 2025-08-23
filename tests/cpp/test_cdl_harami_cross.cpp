#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLHaramiCross) {
  const int N = 3;
  std::vector<float> open{10.0f, 8.7f, 9.0f};
  std::vector<float> high{10.5f, 9.0f, 9.5f};
  std::vector<float> low{7.5f, 8.5f, 8.8f};
  std::vector<float> close{8.0f, 8.69f, 9.2f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_harami_cross(open.data(), high.data(), low.data(),
                                      close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_harami_cross failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
}

