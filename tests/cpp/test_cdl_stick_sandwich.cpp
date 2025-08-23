#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLStickSandwich) {
  const int N = 4;
  std::vector<float> open{10.0f, 9.7f, 10.6f, 10.2f};
  std::vector<float> high{10.2f, 10.5f, 10.7f, 10.6f};
  std::vector<float> low{9.4f, 9.7f, 9.4f, 9.9f};
  std::vector<float> close{9.5f, 10.4f, 9.5f, 10.5f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[0] = std::numeric_limits<float>::quiet_NaN();
  ref[1] = std::numeric_limits<float>::quiet_NaN();
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_stick_sandwich(open.data(), high.data(), low.data(),
                                        close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_stick_sandwich failed";
  expect_approx_equal(out, ref);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
}

