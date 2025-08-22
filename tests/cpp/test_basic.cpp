#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <tacuda.h>
#include <vector>

namespace {

// Helper to compare floating point vectors, ignoring NaNs.
void expect_approx_equal(const std::vector<float> &a,
                         const std::vector<float> &b, float eps = 1e-3f) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i]))
      continue;
    EXPECT_NEAR(a[i], b[i], eps) << "Mismatch at index " << i;
  }
}

std::vector<float> dema_ref(const std::vector<float> &in, int period) {
  std::ostringstream cmd;
  cmd << "python3 - <<'PY'\n";
  cmd << "import numpy as np\n";
  cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n "
         "subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-"
         "Lib'])\n import talib\n";
  cmd << "x=np.array([";
  for (size_t i = 0; i < in.size(); ++i) {
    if (i)
      cmd << ',';
    cmd << in[i];
  }
  cmd << "],dtype=float)\n";
  cmd << "res=talib.DEMA(x,timeperiod=" << period << ")\n";
  cmd << "lb=" << 2 * (period - 1) << "\n";
  cmd << "out=np.full_like(x,float('nan'))\n";
  cmd << "out[:len(x)-lb]=res[lb:]\n";
  cmd << "print('\\n'.join(str(v) for v in out))\n";
  cmd << "PY";
  FILE *pipe = popen(cmd.str().c_str(), "r");
  std::vector<float> out(in.size(), std::numeric_limits<float>::quiet_NaN());
  if (pipe) {
    char buf[128];
    for (size_t i = 0; i < out.size() && fgets(buf, sizeof(buf), pipe); ++i) {
      out[i] = std::strtof(buf, nullptr);
    }
    pclose(pipe);
  }
  return out;
}

std::vector<float> tema_ref(const std::vector<float> &in, int period) {
  std::ostringstream cmd;
  cmd << "python3 - <<'PY'\n";
  cmd << "import numpy as np\n";
  cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n "
         "subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-"
         "Lib'])\n import talib\n";
  cmd << "x=np.array([";
  for (size_t i = 0; i < in.size(); ++i) {
    if (i)
      cmd << ',';
    cmd << in[i];
  }
  cmd << "],dtype=float)\n";
  cmd << "res=talib.TEMA(x,timeperiod=" << period << ")\n";
  cmd << "lb=" << 3 * (period - 1) << "\n";
  cmd << "out=np.full_like(x,float('nan'))\n";
  cmd << "out[:len(x)-lb]=res[lb:]\n";
  cmd << "print('\\n'.join(str(v) for v in out))\n";
  cmd << "PY";
  FILE *pipe = popen(cmd.str().c_str(), "r");
  std::vector<float> out(in.size(), std::numeric_limits<float>::quiet_NaN());
  if (pipe) {
    char buf[128];
    for (size_t i = 0; i < out.size() && fgets(buf, sizeof(buf), pipe); ++i) {
      out[i] = std::strtof(buf, nullptr);
    }
    pclose(pipe);
  }
  return out;
}

std::vector<float> trix_ref(const std::vector<float> &in, int period) {
  std::ostringstream cmd;
  cmd << "python3 - <<'PY'\n";
  cmd << "import numpy as np\n";
  cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n "
         "subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-"
         "Lib'])\n import talib\n";
  cmd << "x=np.array([";
  for (size_t i = 0; i < in.size(); ++i) {
    if (i)
      cmd << ',';
    cmd << in[i];
  }
  cmd << "],dtype=float)\n";
  cmd << "res=talib.TRIX(x,timeperiod=" << period << ")\n";
  cmd << "lb=" << 3 * (period - 1) + 1 << "\n";
  cmd << "out=np.full_like(x,float('nan'))\n";
  cmd << "out[:len(x)-lb]=res[lb:]\n";
  cmd << "print('\\n'.join(str(v) for v in out))\n";
  cmd << "PY";
  FILE *pipe = popen(cmd.str().c_str(), "r");
  std::vector<float> out(in.size(), std::numeric_limits<float>::quiet_NaN());
  if (pipe) {
    char buf[128];
    for (size_t i = 0; i < out.size() && fgets(buf, sizeof(buf), pipe); ++i) {
      out[i] = std::strtof(buf, nullptr);
    }
    pclose(pipe);
  }
  return out;
}

std::vector<float> kama_ref(const std::vector<float> &in, int period,
                            int fastPeriod, int slowPeriod) {
  std::ostringstream cmd;
  cmd << "python3 - <<'PY'\n";
  cmd << "import numpy as np\n";
  cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n "
         "subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-"
         "Lib'])\n import talib\n";
  cmd << "x=np.array([";
  for (size_t i = 0; i < in.size(); ++i) {
    if (i)
      cmd << ',';
    cmd << in[i];
  }
  cmd << "],dtype=float)\n";
  cmd << "res=talib.KAMA(x,timeperiod=" << period
      << ",fastperiod=" << fastPeriod << ",slowperiod=" << slowPeriod << ")\n";
  cmd << "lb=" << period << "\n";
  cmd << "out=np.full_like(x,float('nan'))\n";
  cmd << "out[:len(x)-lb]=res[lb:]\n";
  cmd << "print('\\n'.join(str(v) for v in out))\n";
  cmd << "PY";
  FILE *pipe = popen(cmd.str().c_str(), "r");
  std::vector<float> out(in.size(), std::numeric_limits<float>::quiet_NaN());
  if (pipe) {
    char buf[128];
    for (size_t i = 0; i < out.size() && fgets(buf, sizeof(buf), pipe); ++i) {
      out[i] = std::strtof(buf, nullptr);
    }
    pclose(pipe);
  }
  return out;
}

std::vector<float> sar_ref(const std::vector<float> &high,
                           const std::vector<float> &low, float step,
                           float maxAcc) {
  size_t n = high.size();
  std::vector<float> out(n);
  float af = step;
  float ep = high[0];
  float sar = low[0];
  bool longPos = true;
  out[0] = sar;
  for (size_t i = 1; i < n; ++i) {
    sar = sar + af * (ep - sar);
    if (longPos) {
      sar = std::min(sar, low[i - 1]);
      if (low[i] < sar) {
        longPos = false;
        sar = ep;
        ep = low[i];
        af = step;
        sar = std::max(sar, high[i - 1]);
      } else {
        if (high[i] > ep) {
          ep = high[i];
          af = std::min(af + step, maxAcc);
        }
      }
    } else {
      sar = std::max(sar, high[i - 1]);
      if (high[i] > sar) {
        longPos = true;
        sar = ep;
        ep = high[i];
        af = step;
        sar = std::min(sar, low[i - 1]);
      } else {
        if (low[i] < ep) {
          ep = low[i];
          af = std::min(af + step, maxAcc);
        }
      }
    }
    out[i] = sar;
  }
  return out;
}

} // namespace

TEST(Tacuda, SMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_sma(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sma failed";
  for (int i = 0; i <= N - p; i++) {
    float s = 0.0f;
    for (int k = 0; k < p; k++)
      s += x[i + k];
    ref[i] = s / p;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, WMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_wma(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_wma failed";
  float denom = 0.5f * p * (p + 1);
  for (int i = 0; i <= N - p; i++) {
    float s = 0.0f;
    for (int k = 0; k < p; k++)
      s += x[i + k] * (p - k);
    ref[i] = s / denom;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, EMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_ema(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ema failed";

  const float k = 2.0f / (p + 1.0f);
  for (int i = 0; i <= N - p; ++i) {
    float weight = 1.0f;
    float weightedSum = x[i + p - 1];
    float weightSum = 1.0f;
    for (int j = 1; j < p; ++j) {
      weight *= (1.0f - k);
      weightedSum += x[i + p - 1 - j] * weight;
      weightSum += weight;
    }
    ref[i] = weightedSum / weightSum;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, DEMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_dema(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_dema failed";

  auto ref = dema_ref(x, p);
  expect_approx_equal(out, ref);
  for (int i = N - 2 * p + 2; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, TEMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_tema(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_tema failed";

  auto ref = tema_ref(x, p);
  expect_approx_equal(out, ref);
  for (int i = N - 3 * p + 3; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, TRIX) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_trix(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_trix failed";

  auto ref = trix_ref(x, p);
  expect_approx_equal(out, ref);
  for (int i = N - 3 * p + 2; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, KAMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 10;
  int fastP = 2;
  int slowP = 30;
  ctStatus_t rc = ct_kama(x.data(), out.data(), N, p, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_kama failed";

  auto ref = kama_ref(x, p, fastP, slowP);
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, Momentum) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_momentum(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_momentum failed";
  for (int i = 0; i < N - p; i++)
    ref[i] = x[i + p] - x[i];
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, ROC) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = 1.0f + std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 5;
  ctStatus_t rc = ct_roc(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_roc failed";
  for (int i = 0; i < N - p; ++i)
    ref[i] = (x[i + p] - x[i]) / x[i] * 100.0f;
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, MacdLine) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int fastP = 12, slowP = 26;
  ctStatus_t rc = ct_macd_line(x.data(), out.data(), N, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_macd_line failed";
  for (int i = 0; i < slowP; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
  for (int i = slowP; i < N; i++) {
    EXPECT_TRUE(std::isfinite(out[i])) << "expected finite value at " << i;
  }
}

TEST(Tacuda, RSI) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, 0.0f);

  int p = 14;
  ctStatus_t rc = ct_rsi(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_rsi failed";
  for (int i = 0; i < N - p; ++i) {
    float gain = 0.0f;
    float loss = 0.0f;
    for (int j = 0; j < p; ++j) {
      float diff = x[i + j + 1] - x[i + j];
      if (diff > 0.0f)
        gain += diff;
      else
        loss -= diff;
    }
    float avgGain = gain / p;
    float avgLoss = loss / p;
    float rsi;
    if (avgLoss == 0.0f)
      rsi = (avgGain == 0.0f) ? 50.0f : 100.0f;
    else if (avgGain == 0.0f)
      rsi = 0.0f;
    else {
      float rs = avgGain / avgLoss;
      rsi = 100.0f - 100.0f / (1.0f + rs);
    }
    ref[i] = rsi;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, BBANDS) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> upper(N, 0.0f), middle(N, 0.0f), lower(N, 0.0f);
  std::vector<float> refU(N, 0.0f), refM(N, 0.0f), refL(N, 0.0f);

  int p = 20;
  float up = 2.0f, down = 2.0f;
  ctStatus_t rc = ct_bbands(x.data(), upper.data(), middle.data(), lower.data(),
                            N, p, up, down);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_bbands failed";

  for (int i = 0; i <= N - p; ++i) {
    float sum = 0.0f, sumSq = 0.0f;
    for (int j = 0; j < p; ++j) {
      float v = x[i + j];
      sum += v;
      sumSq += v * v;
    }
    float mean = sum / p;
    float var = sumSq / p - mean * mean;
    var = std::max(var, 0.0f);
    float stddev = std::sqrt(var);
    refM[i] = mean;
    refU[i] = mean + up * stddev;
    refL[i] = mean - down * stddev;
  }

  expect_approx_equal(upper, refU);
  expect_approx_equal(middle, refM);
  expect_approx_equal(lower, refL);

  for (int i = N - p + 1; i < N; ++i) {
    EXPECT_TRUE(std::isnan(upper[i])) << "expected NaN at tail " << i;
    EXPECT_TRUE(std::isnan(middle[i])) << "expected NaN at tail " << i;
    EXPECT_TRUE(std::isnan(lower[i])) << "expected NaN at tail " << i;
  }
}

TEST(Tacuda, ATR) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                            48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                            49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close = {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                              49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                              49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 14;
  ctStatus_t rc =
      ct_atr(high.data(), low.data(), close.data(), out.data(), N, p, 0.0f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_atr failed";

  std::vector<float> tr(N);
  tr[0] = high[0] - low[0];
  for (int i = 1; i < N; ++i) {
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float sum = 0.0f;
  for (int i = 0; i < p; ++i)
    sum += tr[i];
  float atr = sum / p;
  ref[p - 1] = atr;
  for (int i = p; i < N; ++i) {
    atr = (atr * (p - 1) + tr[i]) / p;
    ref[i] = atr;
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, Stochastic) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                            48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                            49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close = {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                              49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                              49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> k(N, 0.0f), d(N, 0.0f);

  int kP = 5, dP = 3;
  ctStatus_t rc = ct_stochastic(high.data(), low.data(), close.data(), k.data(),
                                d.data(), N, kP, dP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_stochastic failed";

  std::vector<float> refK = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             86.45833f,
                             97.29730f,
                             99.40476f,
                             81.93548f,
                             40.60150f,
                             48.12030f,
                             65.89147f,
                             75.19380f,
                             84.24658f};
  std::vector<float> refD = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             89.94871f,
                             93.85261f,
                             94.38680f,
                             92.87918f,
                             73.98058f,
                             56.88576f,
                             51.53776f,
                             63.06852f,
                             75.11062f};

  expect_approx_equal(k, refK);
  expect_approx_equal(d, refD);
  for (int i = 0; i < kP + dP - 2; ++i) {
    EXPECT_TRUE(std::isnan(k[i])) << "expected NaN at head " << i;
    EXPECT_TRUE(std::isnan(d[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, CCI) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                            48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                            49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close = {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                              49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                              49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc =
      ct_cci(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cci failed";

  for (int i = p - 1; i < N; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < p; ++j) {
      int idx = i - j;
      sum += (high[idx] + low[idx] + close[idx]) / 3.0f;
    }
    float sma = sum / p;
    float dev = 0.0f;
    for (int j = 0; j < p; ++j) {
      int idx = i - j;
      float tp = (high[idx] + low[idx] + close[idx]) / 3.0f;
      dev += std::fabs(tp - sma);
    }
    float md = dev / p;
    float tp_cur = (high[i] + low[i] + close[i]) / 3.0f;
    ref[i] = (md == 0.0f) ? 0.0f : (tp_cur - sma) / (0.015f * md);
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, OBV) {
  std::vector<float> price = {1.0f, 2.0f, 2.0f, 1.0f, 3.0f};
  std::vector<float> volume = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  const int N = price.size();
  std::vector<float> out(N, 0.0f), ref = {10.0f, 30.0f, 30.0f, -10.0f, 40.0f};

  ctStatus_t rc = ct_obv(price.data(), volume.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_obv failed";
  expect_approx_equal(out, ref);
}

namespace {
std::vector<float> adosc_ref(const std::vector<float> &high,
                             const std::vector<float> &low,
                             const std::vector<float> &close,
                             const std::vector<float> &volume, int shortP,
                             int longP) {
  int N = high.size();
  std::vector<float> ad(N, 0.0f);
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  float cum = 0.0f;
  for (int i = 0; i < N; ++i) {
    float denom = high[i] - low[i];
    float clv = denom == 0.0f
                    ? 0.0f
                    : ((close[i] - low[i]) - (high[i] - close[i])) / denom;
    cum += clv * volume[i];
    ad[i] = cum;
  }
  auto ema_at = [](const std::vector<float> &x, int idx, int period) {
    const float k = 2.0f / (period + 1.0f);
    float weight = 1.0f;
    float weightedSum = x[idx];
    float weightSum = 1.0f;
    int steps = std::min(period, idx);
    for (int i = 1; i <= steps; ++i) {
      weight *= (1.0f - k);
      weightedSum += x[idx - i] * weight;
      weightSum += weight;
    }
    return weightedSum / weightSum;
  };
  for (int i = longP; i < N; ++i) {
    float emaS = ema_at(ad, i, shortP);
    float emaL = ema_at(ad, i, longP);
    out[i] = emaS - emaL;
  }
  return out;
}
} // namespace

TEST(Tacuda, ADOSC) {
  std::vector<float> high = {12.f, 12.5f, 13.f, 13.5f, 14.f, 14.5f};
  std::vector<float> low = {11.f, 11.5f, 12.f, 12.5f, 13.f, 13.5f};
  std::vector<float> close = {11.5f, 12.f, 12.5f, 13.f, 13.5f, 14.f};
  std::vector<float> volume = {100.f, 110.f, 120.f, 130.f, 140.f, 150.f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int shortP = 3, longP = 5;
  ctStatus_t rc = ct_adosc(high.data(), low.data(), close.data(), volume.data(),
                           out.data(), N, shortP, longP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_adosc failed";
  auto ref = adosc_ref(high, low, close, volume, shortP, longP);
  expect_approx_equal(out, ref);
  for (int i = 0; i < longP; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

namespace {
std::vector<float> ultosc_ref(const std::vector<float> &high,
                              const std::vector<float> &low,
                              const std::vector<float> &close, int shortP,
                              int medP, int longP) {
  int N = high.size();
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  for (int idx = longP; idx < N; ++idx) {
    float bp1 = 0.0f, tr1 = 0.0f;
    float bp2 = 0.0f, tr2 = 0.0f;
    float bp3 = 0.0f, tr3 = 0.0f;
    for (int j = 0; j < longP; ++j) {
      int i = idx - j;
      float prev = (i > 0) ? close[i - 1] : close[i];
      float bp = close[i] - std::fmin(low[i], prev);
      float tr = std::fmax(high[i], prev) - std::fmin(low[i], prev);
      if (j < shortP) {
        bp1 += bp;
        tr1 += tr;
      }
      if (j < medP) {
        bp2 += bp;
        tr2 += tr;
      }
      bp3 += bp;
      tr3 += tr;
    }
    float avg1 = (tr1 == 0.0f) ? 0.0f : bp1 / tr1;
    float avg2 = (tr2 == 0.0f) ? 0.0f : bp2 / tr2;
    float avg3 = (tr3 == 0.0f) ? 0.0f : bp3 / tr3;
    out[idx] = 100.0f * (4.0f * avg1 + 2.0f * avg2 + avg3) / 7.0f;
  }
  return out;
}
} // namespace

TEST(Tacuda, ULTOSC) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int sp = 3, mp = 5, lp = 7;
  ctStatus_t rc =
      ct_ultosc(high.data(), low.data(), close.data(), out.data(), N, sp, mp, lp);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ultosc failed";
  auto ref = ultosc_ref(high, low, close, sp, mp, lp);
  expect_approx_equal(out, ref);
  for (int i = 0; i < lp; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, ADX) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 3;
  ctStatus_t rc =
      ct_adx(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_adx failed";

  std::vector<float> dmp(N, 0.0f), dmm(N, 0.0f), tr(N, 0.0f), dx(N, 0.0f);
  for (int i = 1; i < N; ++i) {
    float upMove = high[i] - high[i - 1];
    float downMove = low[i - 1] - low[i];
    dmp[i] = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
    dmm[i] = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float dmp_s = 0.0f, dmm_s = 0.0f, tr_s = 0.0f;
  for (int i = 1; i <= p; ++i) {
    dmp_s += dmp[i];
    dmm_s += dmm[i];
    tr_s += tr[i];
  }
  float dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
  float dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
  dx[p] =
      (dip + dim == 0.0f) ? 0.0f : 100.0f * std::fabs(dip - dim) / (dip + dim);
  float dx_sum = dx[p];
  for (int i = p + 1; i < N; ++i) {
    dmp_s = dmp_s - dmp_s / p + dmp[i];
    dmm_s = dmm_s - dmm_s / p + dmm[i];
    tr_s = tr_s - tr_s / p + tr[i];
    dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
    dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
    dx[i] = (dip + dim == 0.0f) ? 0.0f
                                : 100.0f * std::fabs(dip - dim) / (dip + dim);
    if (i < 2 * p) {
      dx_sum += dx[i];
      if (i == 2 * p - 1)
        ref[i] = dx_sum / p;
    } else {
      ref[i] = ((ref[i - 1] * (p - 1)) + dx[i]) / p;
    }
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < 2 * p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}

TEST(Tacuda, SARTrending) {
  std::vector<float> high = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sar(high.data(), low.data(), out.data(), N, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sar failed";
  auto ref = sar_ref(high, low, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, SARRanging) {
  std::vector<float> high = {5.f, 6.f, 5.5f, 6.2f, 5.8f, 6.4f, 5.9f, 6.5f};
  std::vector<float> low = {4.f, 5.f, 4.5f, 5.2f, 4.8f, 5.4f, 4.9f, 5.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_sar(high.data(), low.data(), out.data(), N, 0.02f, 0.2f);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_sar failed";
  auto ref = sar_ref(high, low, 0.02f, 0.2f);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, Aroon) {
  std::vector<float> high = {1.f, 2.f, 3.f, 2.f, 3.f, 4.f, 5.f, 4.f, 6.f, 7.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 1.5f, 2.5f,
                            3.5f, 4.5f, 3.5f, 5.5f, 6.5f};
  const int N = high.size();
  std::vector<float> up(N, 0.0f), down(N, 0.0f), osc(N, 0.0f);
  int pUp = 5, pDown = 5;
  ctStatus_t rc = ct_aroon(high.data(), low.data(), up.data(), down.data(),
                           osc.data(), N, pUp, pDown);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_aroon failed";

  std::vector<float> refUp(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refDown(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = pUp; i < N; ++i) {
    int sinceHigh = 0;
    float maxVal = high[i];
    for (int j = 1; j <= pUp; ++j) {
      float val = high[i - j];
      if (val >= maxVal) {
        maxVal = val;
        sinceHigh = j;
      }
    }
    refUp[i] = 100.0f * (pUp - sinceHigh) / pUp;
  }
  for (int i = pDown; i < N; ++i) {
    int sinceLow = 0;
    float minVal = low[i];
    for (int j = 1; j <= pDown; ++j) {
      float val = low[i - j];
      if (val <= minVal) {
        minVal = val;
        sinceLow = j;
      }
    }
    refDown[i] = 100.0f * (pDown - sinceLow) / pDown;
  }
  std::vector<float> refOsc(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = std::max(pUp, pDown); i < N; ++i)
    refOsc[i] = refUp[i] - refDown[i];

  expect_approx_equal(up, refUp);
  expect_approx_equal(down, refDown);
  expect_approx_equal(osc, refOsc);

  for (int i = 0; i < pUp; ++i)
    EXPECT_TRUE(std::isnan(up[i])) << "expected NaN at head " << i;
  for (int i = 0; i < pDown; ++i)
    EXPECT_TRUE(std::isnan(down[i])) << "expected NaN at head " << i;
  for (int i = 0; i < std::max(pUp, pDown); ++i)
    EXPECT_TRUE(std::isnan(osc[i])) << "expected NaN at head " << i;
}

TEST(Tacuda, MFI) {
  const int N = 30;
  const int p = 14;
  std::vector<float> high = {
      103.1797f, 103.1869f, 103.6822f, 106.0394f, 108.2978f, 106.3645f,
      108.2244f, 108.0788f, 107.2852f, 107.5736f, 107.9974f, 109.5241f,
      110.5949f, 110.5192f, 111.7877f, 110.7919f, 112.4462f, 112.1697f,
      113.2205f, 111.7666f, 109.5332f, 109.854f,  110.5902f, 109.7752f,
      112.8638f, 110.6323f, 110.7656f, 110.8366f, 113.0478f, 113.4313f};
  std::vector<float> low = {
      100.5071f, 102.0201f, 101.6783f, 104.6809f, 105.7863f, 105.3669f,
      106.1153f, 107.0141f, 106.5454f, 107.1999f, 107.0801f, 108.8005f,
      109.2626f, 109.2399f, 110.2089f, 109.6001f, 111.283f,  111.5296f,
      111.4559f, 111.2458f, 107.9698f, 108.0934f, 109.8739f, 108.6085f,
      111.6816f, 109.3505f, 110.0366f, 110.0088f, 110.9366f, 113.2555f};
  std::vector<float> close = {
      101.1815f, 102.6146f, 103.3758f, 104.6157f, 107.9955f, 107.2221f,
      107.8136f, 106.9829f, 106.4343f, 107.9075f, 107.3227f, 109.5898f,
      109.8437f, 110.3496f, 110.4833f, 110.9921f, 112.1381f, 112.8207f,
      112.3042f, 111.5877f, 109.7753f, 108.8134f, 109.7165f, 110.0943f,
      111.2928f, 111.3968f, 110.2639f, 109.9098f, 112.7778f, 114.026f};
  std::vector<float> volume = {
      835.f, 986.f, 912.f, 316.f, 481.f, 124.f, 167.f, 871.f, 334.f, 816.f,
      391.f, 826.f, 801.f, 809.f, 827.f, 655.f, 132.f, 111.f, 716.f, 312.f,
      238.f, 794.f, 521.f, 737.f, 768.f, 723.f, 588.f, 870.f, 639.f, 317.f};
  std::vector<float> out(N, 0.0f);

  ctStatus_t rc = ct_mfi(high.data(), low.data(), close.data(), volume.data(),
                         out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_mfi failed";

  std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
  float expected[] = {80.1529f, 71.4794f, 68.7325f, 67.9212f,
                      69.0350f, 67.2809f, 64.5633f, 65.0960f,
                      70.0452f, 60.4856f, 67.0099f, 57.7525f,
                      49.2357f, 38.7765f, 37.3642f, 43.3576f};
  for (int i = 0; i < 16; ++i) {
    ref[p + i] = expected[i];
  }

  expect_approx_equal(out, ref);
  for (int i = 0; i < p; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}

namespace {

std::vector<float> ad_ref(const std::vector<float> &high,
                          const std::vector<float> &low,
                          const std::vector<float> &close,
                          const std::vector<float> &volume) {
  int N = high.size();
  std::vector<float> out(N, 0.0f);
  float cum = 0.0f;
  for (int i = 0; i < N; ++i) {
    float denom = high[i] - low[i];
    float clv = denom == 0.0f
                    ? 0.0f
                    : ((close[i] - low[i]) - (high[i] - close[i])) / denom;
    cum += clv * volume[i];
    out[i] = cum;
  }
  return out;
}

float ema_at_ref(const std::vector<float> &x, int idx, int period) {
  const float k = 2.0f / (period + 1.0f);
  float weight = 1.0f;
  float weightedSum = x[idx];
  float weightSum = 1.0f;
  int steps = std::min(period, idx);
  for (int i = 1; i <= steps; ++i) {
    weight *= (1.0f - k);
    weightedSum += x[idx - i] * weight;
    weightSum += weight;
  }
  return weightedSum / weightSum;
}

std::vector<float> apo_ref(const std::vector<float> &in, int fastP,
                           int slowP) {
  int N = in.size();
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = slowP; i < N; ++i) {
    float emaFast = ema_at_ref(in, i, fastP);
    float emaSlow = ema_at_ref(in, i, slowP);
    out[i] = emaFast - emaSlow;
  }
  return out;
}

std::vector<float> aroonosc_ref(const std::vector<float> &high,
                                const std::vector<float> &low, int period) {
  int N = high.size();
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = period; i < N; ++i) {
    int sinceHigh = 0, sinceLow = 0;
    float maxVal = high[i];
    float minVal = low[i];
    for (int j = 1; j <= period; ++j) {
      float h = high[i - j];
      float l = low[i - j];
      if (h >= maxVal) {
        maxVal = h;
        sinceHigh = j;
      }
      if (l <= minVal) {
        minVal = l;
        sinceLow = j;
      }
    }
    float up = 100.0f * (period - sinceHigh) / period;
    float down = 100.0f * (period - sinceLow) / period;
    out[i] = up - down;
  }
  return out;
}

std::vector<float> adxr_ref(const std::vector<float> &high,
                            const std::vector<float> &low,
                            const std::vector<float> &close, int p) {
  int N = high.size();
  std::vector<float> adx(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> dmp(N, 0.0f), dmm(N, 0.0f), tr(N, 0.0f), dx(N, 0.0f);
  for (int i = 1; i < N; ++i) {
    float upMove = high[i] - high[i - 1];
    float downMove = low[i - 1] - low[i];
    dmp[i] = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
    dmm[i] = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float dmp_s = 0.0f, dmm_s = 0.0f, tr_s = 0.0f;
  for (int i = 1; i <= p; ++i) {
    dmp_s += dmp[i];
    dmm_s += dmm[i];
    tr_s += tr[i];
  }
  float dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
  float dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
  dx[p] =
      (dip + dim == 0.0f) ? 0.0f : 100.0f * std::fabs(dip - dim) / (dip + dim);
  float dx_sum = dx[p];
  for (int i = p + 1; i < N; ++i) {
    dmp_s = dmp_s - dmp_s / p + dmp[i];
    dmm_s = dmm_s - dmm_s / p + dmm[i];
    tr_s = tr_s - tr_s / p + tr[i];
    dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
    dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
    dx[i] = (dip + dim == 0.0f)
                ? 0.0f
                : 100.0f * std::fabs(dip - dim) / (dip + dim);
    if (i < 2 * p) {
      dx_sum += dx[i];
      if (i == 2 * p - 1)
        adx[i] = dx_sum / p;
    } else {
      adx[i] = ((adx[i - 1] * (p - 1)) + dx[i]) / p;
    }
  }
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = 3 * p - 1; i < N; ++i) {
    out[i] = 0.5f * (adx[i] + adx[i - p]);
  }
  return out;
}

std::vector<float> avgprice_ref(const std::vector<float> &open,
                                const std::vector<float> &high,
                                const std::vector<float> &low,
                                const std::vector<float> &close) {
  int N = open.size();
  std::vector<float> out(N, 0.0f);
  for (int i = 0; i < N; ++i)
    out[i] = (open[i] + high[i] + low[i] + close[i]) * 0.25f;
  return out;
}

} // namespace

TEST(Tacuda, AD) {
  std::vector<float> high = {12.f, 12.5f, 13.f, 13.5f};
  std::vector<float> low = {11.f, 11.5f, 12.f, 12.5f};
  std::vector<float> close = {11.5f, 12.f, 12.5f, 13.f};
  std::vector<float> volume = {100.f, 110.f, 120.f, 130.f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc =
      ct_ad(high.data(), low.data(), close.data(), volume.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ad failed";
  auto ref = ad_ref(high, low, close, volume);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, ADXR) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int p = 3;
  ctStatus_t rc =
      ct_adxr(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_adxr failed";
  auto ref = adxr_ref(high, low, close, p);
  expect_approx_equal(out, ref);
  for (int i = 0; i < 3 * p - 1; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}

TEST(Tacuda, APO) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f);
  int fastP = 12, slowP = 26;
  ctStatus_t rc = ct_apo(x.data(), out.data(), N, fastP, slowP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_apo failed";
  auto ref = apo_ref(x, fastP, slowP);
  expect_approx_equal(out, ref);
  for (int i = 0; i < slowP; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}

TEST(Tacuda, AroonOscillator) {
  std::vector<float> high = {1.f, 2.f, 3.f, 2.f, 3.f, 4.f, 5.f, 4.f, 6.f, 7.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 1.5f, 2.5f,
                            3.5f, 4.5f, 3.5f, 5.5f, 6.5f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int p = 5;
  ctStatus_t rc =
      ct_aroonosc(high.data(), low.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_aroonosc failed";
  auto ref = aroonosc_ref(high, low, p);
  expect_approx_equal(out, ref);
  for (int i = 0; i < p; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}

TEST(Tacuda, AvgPrice) {
  std::vector<float> open = {10.f, 11.f, 12.f};
  std::vector<float> high = {12.f, 13.f, 14.f};
  std::vector<float> low = {9.f, 10.f, 11.f};
  std::vector<float> close = {11.f, 12.f, 13.f};
  const int N = open.size();
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_avgprice(open.data(), high.data(), low.data(),
                              close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_avgprice failed";
  auto ref = avgprice_ref(open, high, low, close);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, Beta) {
  const int N = 128;
  std::vector<float> x(N), y(N);
  for (int i = 0; i < N; ++i) {
    x[i] = std::sin(0.01f * i);
    y[i] = std::cos(0.02f * i);
  }
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_beta(x.data(), y.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_beta failed";
  for (int i = 0; i < N - p; ++i) {
    float sumX = 0.0f, sumY = 0.0f;
    for (int j = 0; j < p; ++j) {
      sumX += x[i + j];
      sumY += y[i + j];
    }
    float meanX = sumX / p;
    float meanY = sumY / p;
    float cov = 0.0f, varY = 0.0f;
    for (int j = 0; j < p; ++j) {
      float dx = x[i + j] - meanX;
      float dy = y[i + j] - meanY;
      cov += dx * dy;
      varY += dy * dy;
    }
    ref[i] = (varY == 0.0f) ? 0.0f : cov / varY;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}

TEST(Tacuda, BOP) {
  const int N = 64;
  std::vector<float> open(N), high(N), low(N), close(N);
  for (int i = 0; i < N; ++i) {
    open[i] = 10.0f + 0.1f * i;
    high[i] = open[i] + 1.0f;
    low[i] = open[i] - 1.0f;
    close[i] = open[i] + std::sin(0.1f * i);
  }
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  ctStatus_t rc = ct_bop(open.data(), high.data(), low.data(), close.data(),
                         out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_bop failed";
  for (int i = 0; i < N; ++i) {
    float denom = high[i] - low[i];
    ref[i] = (denom == 0.0f) ? 0.0f : (close[i] - open[i]) / denom;
  }
  expect_approx_equal(out, ref);
}

TEST(Tacuda, CMO) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_cmo(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cmo failed";
  for (int i = 0; i < N - p; ++i) {
    float up = 0.0f, down = 0.0f;
    for (int j = 0; j < p; ++j) {
      float diff = x[i + j + 1] - x[i + j];
      if (diff > 0.0f)
        up += diff;
      else
        down -= diff;
    }
    float denom = up + down;
    ref[i] = (denom == 0.0f) ? 0.0f : 100.0f * (up - down) / denom;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}

TEST(Tacuda, Correl) {
  const int N = 128;
  std::vector<float> x(N), y(N);
  for (int i = 0; i < N; ++i) {
    x[i] = std::sin(0.01f * i);
    y[i] = std::cos(0.03f * i);
  }
  std::vector<float> out(N, 0.0f), ref(N, 0.0f);
  int p = 5;
  ctStatus_t rc = ct_correl(x.data(), y.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_correl failed";
  for (int i = 0; i < N - p; ++i) {
    float sumX = 0.0f, sumY = 0.0f;
    for (int j = 0; j < p; ++j) {
      sumX += x[i + j];
      sumY += y[i + j];
    }
    float meanX = sumX / p;
    float meanY = sumY / p;
    float cov = 0.0f, varX = 0.0f, varY = 0.0f;
    for (int j = 0; j < p; ++j) {
      float dx = x[i + j] - meanX;
      float dy = y[i + j] - meanY;
      cov += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }
    float denom = std::sqrt(varX * varY);
    ref[i] = (denom == 0.0f) ? 0.0f : cov / denom;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}

TEST(Tacuda, DX) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 3;
  ctStatus_t rc = ct_dx(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_dx failed";
  for (int i = 0; i < N - p; ++i) {
    float prevHigh = high[i];
    float prevLow = low[i];
    float prevClose = close[i];
    float dmp = 0.0f, dmm = 0.0f, tr = 0.0f;
    for (int j = 1; j <= p; ++j) {
      float curHigh = high[i + j];
      float curLow = low[i + j];
      float upMove = curHigh - prevHigh;
      float downMove = prevLow - curLow;
      float dmPlus = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
      float dmMinus = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
      float trVal = std::max(curHigh - curLow,
                              std::max(std::fabs(curHigh - prevClose),
                                       std::fabs(curLow - prevClose)));
      dmp += dmPlus;
      dmm += dmMinus;
      tr += trVal;
      prevHigh = curHigh;
      prevLow = curLow;
      prevClose = close[i + j];
    }
    float dip = (tr == 0.0f) ? 0.0f : 100.0f * dmp / tr;
    float dim = (tr == 0.0f) ? 0.0f : 100.0f * dmm / tr;
    float denom = dip + dim;
    ref[i] = (denom == 0.0f) ? 0.0f : 100.0f * std::fabs(dip - dim) / denom;
  }
  expect_approx_equal(out, ref);
  for (int i = N - p; i < N; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at tail " << i;
}

