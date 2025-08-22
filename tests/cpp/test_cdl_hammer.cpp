#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLHammer) {
  const int N = 4;
  std::vector<float> open{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> high{1.3f, 2.08f, 3.5f, 4.3f};
  std::vector<float> low{0.9f, 1.5f, 2.9f, 3.5f};
  std::vector<float> close{1.2f, 2.05f, 3.4f, 3.9f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[1] = 1.0f;
  ctStatus_t rc = ct_cdl_hammer(open.data(), high.data(), low.data(), close.data(),
                                out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_hammer failed";
  expect_approx_equal(out, ref);
}
