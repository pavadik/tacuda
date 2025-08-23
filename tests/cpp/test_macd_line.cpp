#include <tacuda.h>
#include "test_utils.hpp"
#include <cuda_runtime.h>

TEST(Tacuda, MACD_LineLargePeriod) {
  const int N = 200000;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.001f * i);

  int fastP = 5000, slowP = 10000;
  std::vector<float> out(N, 0.0f);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  ctStatus_t rc = ct_macd_line(x.data(), out.data(), N, fastP, slowP);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  EXPECT_LT(ms, 200.0f);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);

  std::vector<float> emaFast(N), emaSlow(N), ref(N, std::numeric_limits<float>::quiet_NaN());
  float kFast = 2.0f / (fastP + 1.0f);
  float kSlow = 2.0f / (slowP + 1.0f);
  emaFast[0] = emaSlow[0] = x[0];
  for (int i = 1; i < N; ++i) {
    emaFast[i] = kFast * x[i] + (1.0f - kFast) * emaFast[i - 1];
    emaSlow[i] = kSlow * x[i] + (1.0f - kSlow) * emaSlow[i - 1];
  }
  for (int i = slowP; i < N; ++i)
    ref[i] = emaFast[i] - emaSlow[i];

  expect_approx_equal(out, ref, 1e-3f);
  for (int i = 0; i < slowP; ++i)
    EXPECT_TRUE(std::isnan(out[i]));
}

