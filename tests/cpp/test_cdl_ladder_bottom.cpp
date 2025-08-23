#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLLadderBottom) {
  const int N = 5;
  std::vector<float> open{5.0f, 4.0f, 3.0f, 2.4f, 2.2f};
  std::vector<float> high{5.2f, 4.3f, 3.2f, 2.5f, 3.8f};
  std::vector<float> low{3.8f, 2.9f, 1.9f, 2.0f, 2.0f};
  std::vector<float> close{4.0f, 3.0f, 2.0f, 2.1f, 3.5f};
  std::vector<float> out(N), ref(N, 0.0f);
  ref[4] = 1.0f;
  ctStatus_t rc = ct_cdl_ladder_bottom(open.data(), high.data(), low.data(),
                                       close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_ladder_bottom failed";
  expect_approx_equal(out, ref);
}

