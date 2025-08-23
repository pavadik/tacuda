#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLHomingPigeon) {
  const int N = 3;
  std::vector<float> open{10.0f, 9.5f, 9.7f};
  std::vector<float> high{10.5f, 9.8f, 9.9f};
  std::vector<float> low{7.5f, 8.9f, 9.4f};
  std::vector<float> close{8.0f, 8.5f, 9.6f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_homing_pigeon(open.data(), high.data(), low.data(),
                                       close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_homing_pigeon failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
}

