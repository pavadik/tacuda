#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLLongLeggedDoji) {
  const int N = 2;
  std::vector<float> open{1.0f, 10.0f};
  std::vector<float> high{1.2f, 12.0f};
  std::vector<float> low{0.8f, 8.0f};
  std::vector<float> close{1.1f, 10.01f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_long_legged_doji(open.data(), high.data(), low.data(),
                                          close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_long_legged_doji failed";
  expect_approx_equal(out, ref);
}

