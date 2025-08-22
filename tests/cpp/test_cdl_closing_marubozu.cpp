#include "test_utils.hpp"
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLClosingMarubozu) {
  const int N = 5;
  std::vector<float> open{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> high{1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
  std::vector<float> low{0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
  std::vector<float> close{1.2f, 2.3f, 3.5f, 3.8f, 4.8f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[2] = 1.0f;
  ctStatus_t rc = ct_cdl_closing_marubozu(open.data(), high.data(), low.data(),
                                          close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_closing_marubozu failed";
  expect_approx_equal(out, ref);
}
