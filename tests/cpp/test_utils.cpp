#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

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

std::vector<float> ht_dcperiod_ref(const std::vector<float> &in) {
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
  cmd << "res=talib.HT_DCPERIOD(x)\n";
  cmd << "print('\\n'.join(str(v) for v in res))\n";
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

std::vector<float> ht_dcphase_ref(const std::vector<float> &in) {
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
  cmd << "res=talib.HT_DCPHASE(x)\n";
  cmd << "print('\\n'.join(str(v) for v in res))\n";
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

std::pair<std::vector<float>, std::vector<float>>
ht_phasor_ref(const std::vector<float> &in) {
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
  cmd << "r1,r2=talib.HT_PHASOR(x)\n";
  cmd << "print('\\n'.join(str(v) for v in r1))\n";
  cmd << "print('---')\n";
  cmd << "print('\\n'.join(str(v) for v in r2))\n";
  cmd << "PY";
  FILE *pipe = popen(cmd.str().c_str(), "r");
  std::vector<float> a(in.size(), std::numeric_limits<float>::quiet_NaN()),
      b(in.size(), std::numeric_limits<float>::quiet_NaN());
  if (pipe) {
    char buf[128];
    size_t idx = 0;
    bool second = false;
    while (fgets(buf, sizeof(buf), pipe)) {
      if (strncmp(buf, "---", 3) == 0) {
        second = true;
        idx = 0;
        continue;
      }
      float v = strtof(buf, nullptr);
      if (!second) {
        if (idx < a.size())
          a[idx++] = v;
      } else {
        if (idx < b.size())
          b[idx++] = v;
      }
    }
    pclose(pipe);
  }
  return {a, b};
}

std::pair<std::vector<float>, std::vector<float>>
ht_sine_ref(const std::vector<float> &in) {
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
  cmd << "r1,r2=talib.HT_SINE(x)\n";
  cmd << "print('\\n'.join(str(v) for v in r1))\n";
  cmd << "print('---')\n";
  cmd << "print('\\n'.join(str(v) for v in r2))\n";
  cmd << "PY";
  FILE *pipe = popen(cmd.str().c_str(), "r");
  std::vector<float> a(in.size(), std::numeric_limits<float>::quiet_NaN()),
      b(in.size(), std::numeric_limits<float>::quiet_NaN());
  if (pipe) {
    char buf[128];
    size_t idx = 0;
    bool second = false;
    while (fgets(buf, sizeof(buf), pipe)) {
      if (strncmp(buf, "---", 3) == 0) {
        second = true;
        idx = 0;
        continue;
      }
      float v = strtof(buf, nullptr);
      if (!second) {
        if (idx < a.size())
          a[idx++] = v;
      } else {
        if (idx < b.size())
          b[idx++] = v;
      }
    }
    pclose(pipe);
  }
  return {a, b};
}

std::vector<float> ht_trendmode_ref(const std::vector<float> &in) {
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
  cmd << "res=talib.HT_TRENDMODE(x)\n";
  cmd << "print('\\n'.join(str(v) for v in res))\n";
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

std::vector<float> ht_trendline_ref(const std::vector<float> &in) {
  std::ostringstream cmd;
  cmd << "python3 - <<'PY'\n";
  cmd << "import numpy as np\n";
  cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n "
          "subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-Lib'])\n import talib\n";
  cmd << "x=np.array([";
  for (size_t i = 0; i < in.size(); ++i) {
    if (i)
      cmd << ',';
    cmd << in[i];
  }
  cmd << "],dtype=float)\n";
  cmd << "res=talib.HT_TRENDLINE(x)\n";
  cmd << "print('\\n'.join(str(v) for v in res))\n";
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

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
macdfix_ref(const std::vector<float> &in, int signalPeriod) {
  std::ostringstream cmd;
  cmd << "python3 - <<'PY'\n";
  cmd << "import numpy as np\n";
  cmd << "try:\n import talib\nexcept Exception:\n import subprocess, sys\n "
          "subprocess.check_call([sys.executable,'-m','pip','install','-q','TA-Lib'])\n import talib\n";
  cmd << "x=np.array([";
  for (size_t i = 0; i < in.size(); ++i) {
    if (i)
      cmd << ',';
    cmd << in[i];
  }
  cmd << "],dtype=float)\n";
  cmd << "macd,signal,hist=talib.MACDFIX(x,signalperiod=" << signalPeriod << ")\n";
  cmd << "print('\\n'.join(str(v) for v in macd))\n";
  cmd << "print('@@')\n";
  cmd << "print('\\n'.join(str(v) for v in signal))\n";
  cmd << "print('@@')\n";
  cmd << "print('\\n'.join(str(v) for v in hist))\n";
  cmd << "PY";
  FILE *pipe = popen(cmd.str().c_str(), "r");
  std::vector<float> macd(in.size(), std::numeric_limits<float>::quiet_NaN());
  std::vector<float> signal(in.size(), std::numeric_limits<float>::quiet_NaN());
  std::vector<float> hist(in.size(), std::numeric_limits<float>::quiet_NaN());
  if (pipe) {
    char buf[256];
    int section = 0;
    size_t idx = 0;
    while (fgets(buf, sizeof(buf), pipe)) {
      if (std::strcmp(buf, "@@\n") == 0) {
        section++;
        idx = 0;
        continue;
      }
      float v = std::strtof(buf, nullptr);
      if (section == 0 && idx < macd.size())
        macd[idx++] = v;
      else if (section == 1 && idx < signal.size())
        signal[idx++] = v;
      else if (section == 2 && idx < hist.size())
        hist[idx++] = v;
    }
    pclose(pipe);
  }
  return {macd, signal, hist};
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
        std::vector<float> adosc_ref(
            const std::vector<float> &high, const std::vector<float> &low,
            const std::vector<float> &close, const std::vector<float> &volume,
            int shortP, int longP) {
          int N = high.size();
          std::vector<float> ad(N, 0.0f);
          std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
          float cum = 0.0f;
          for (int i = 0; i < N; ++i) {
            float denom = high[i] - low[i];
            float clv =
                denom == 0.0f
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
        std::vector<float> ultosc_ref(
            const std::vector<float> &high, const std::vector<float> &low,
            const std::vector<float> &close, int shortP, int medP, int longP) {
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
        std::vector<float> ad_ref(
            const std::vector<float> &high, const std::vector<float> &low,
            const std::vector<float> &close, const std::vector<float> &volume) {
          int N = high.size();
          std::vector<float> out(N, 0.0f);
          float cum = 0.0f;
          for (int i = 0; i < N; ++i) {
            float denom = high[i] - low[i];
            float clv =
                denom == 0.0f
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

        std::vector<float> ppo_ref(const std::vector<float> &in, int fastP,
                                   int slowP) {
          int N = in.size();
          std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
          for (int i = slowP; i < N; ++i) {
            float emaFast = ema_at_ref(in, i, fastP);
            float emaSlow = ema_at_ref(in, i, slowP);
            out[i] = (emaSlow == 0.0f) ? 0.0f
                                       : 100.0f * (emaFast - emaSlow) / emaSlow;
          }
          return out;
        }

        std::vector<float> pvo_ref(const std::vector<float> &in, int fastP,
                                   int slowP) {
          return ppo_ref(in, fastP, slowP);
        }

        std::vector<float> aroonosc_ref(const std::vector<float> &high,
                                        const std::vector<float> &low,
                                        int period) {
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
          std::vector<float> dmp(N, 0.0f), dmm(N, 0.0f), tr(N, 0.0f),
              dx(N, 0.0f);
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
          dx[p] = (dip + dim == 0.0f)
                      ? 0.0f
                      : 100.0f * std::fabs(dip - dim) / (dip + dim);
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

        std::vector<float> avgprice_ref(
            const std::vector<float> &open, const std::vector<float> &high,
            const std::vector<float> &low, const std::vector<float> &close) {
          int N = open.size();
          std::vector<float> out(N, 0.0f);
          for (int i = 0; i < N; ++i)
            out[i] = (open[i] + high[i] + low[i] + close[i]) * 0.25f;
          return out;
        }

        std::vector<float> typprice_ref(const std::vector<float> &high,
                                        const std::vector<float> &low,
                                        const std::vector<float> &close) {
          int N = high.size();
          std::vector<float> out(N, 0.0f);
          for (int i = 0; i < N; ++i)
            out[i] = (high[i] + low[i] + close[i]) / 3.0f;
          return out;
        }

        std::vector<float> wclprice_ref(const std::vector<float> &high,
                                        const std::vector<float> &low,
                                        const std::vector<float> &close) {
          int N = high.size();
          std::vector<float> out(N, 0.0f);
          for (int i = 0; i < N; ++i)
            out[i] = (high[i] + low[i] + 2.0f * close[i]) * 0.25f;
          return out;
        }

        static std::vector<float> run_linearreg_python(
            const std::vector<float> &in, int period, const char *func) {
          std::ostringstream cmd;
          cmd << "python3 - <<'PY'\n";
          cmd << "import numpy as np\n";
          cmd << "try:\n import talib\nexcept Exception:\n import subprocess, "
                 "sys\n "
                 "subprocess.check_call([sys.executable,'-m','pip','install','-"
                 "q','TA-Lib'])\n import talib\n";
          cmd << "x=np.array([";
          for (size_t i = 0; i < in.size(); ++i) {
            if (i)
              cmd << ',';
            cmd << in[i];
          }
          cmd << "],dtype=float)\n";
          cmd << "res=getattr(talib,'" << func << "')(x,timeperiod=" << period
              << ")\n";
          cmd << "lb=" << (period - 1) << "\n";
          cmd << "out=np.full_like(x,float('nan'))\n";
          cmd << "out[:len(x)-lb]=res[lb:]\n";
          cmd << "print('\\n'.join(str(v) for v in out))\n";
          cmd << "PY";
          FILE *pipe = popen(cmd.str().c_str(), "r");
          std::vector<float> out(in.size(),
                                 std::numeric_limits<float>::quiet_NaN());
          if (pipe) {
            char buf[128];
            for (size_t i = 0; i < out.size() && fgets(buf, sizeof(buf), pipe);
                 ++i) {
              out[i] = std::strtof(buf, nullptr);
            }
            pclose(pipe);
          }
          return out;
        }

        std::vector<float> linearreg_ref(const std::vector<float> &in,
                                         int period) {
          return run_linearreg_python(in, period, "LINEARREG");
        }

        std::vector<float> linearreg_slope_ref(const std::vector<float> &in,
                                               int period) {
          return run_linearreg_python(in, period, "LINEARREG_SLOPE");
        }

        std::vector<float> linearreg_intercept_ref(const std::vector<float> &in,
                                                   int period) {
          return run_linearreg_python(in, period, "LINEARREG_INTERCEPT");
        }

        std::vector<float> linearreg_angle_ref(const std::vector<float> &in,
                                               int period) {
          return run_linearreg_python(in, period, "LINEARREG_ANGLE");
        }

        std::vector<float> tsf_ref(const std::vector<float> &in, int period) {
          return run_linearreg_python(in, period, "TSF");
        }

        std::vector<float> plus_dm_ref(const std::vector<float> &high,
                                       const std::vector<float> &low,
                                       int period) {
          int N = high.size();
          std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
          float prevHigh = high[0];
          float prevLow = low[0];
          float prevDM = 0.0f;
          for (int i = 1; i < period; ++i) {
            float diffP = high[i] - prevHigh;
            prevHigh = high[i];
            float diffM = prevLow - low[i];
            prevLow = low[i];
            if (diffP > 0.0f && diffP > diffM)
              prevDM += diffP;
          }
          out[period - 1] = prevDM;
          for (int i = period; i < N; ++i) {
            float diffP = high[i] - prevHigh;
            prevHigh = high[i];
            float diffM = prevLow - low[i];
            prevLow = low[i];
            if (diffP > 0.0f && diffP > diffM)
              prevDM = prevDM - prevDM / period + diffP;
            else
              prevDM = prevDM - prevDM / period;
            out[i] = prevDM;
          }
          return out;
        }

        std::vector<float> minus_dm_ref(const std::vector<float> &high,
                                        const std::vector<float> &low,
                                        int period) {
          int N = high.size();
          std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
          float prevHigh = high[0];
          float prevLow = low[0];
          float prevDM = 0.0f;
          for (int i = 1; i < period; ++i) {
            float diffP = high[i] - prevHigh;
            prevHigh = high[i];
            float diffM = prevLow - low[i];
            prevLow = low[i];
            if (diffM > 0.0f && diffM > diffP)
              prevDM += diffM;
          }
          out[period - 1] = prevDM;
          for (int i = period; i < N; ++i) {
            float diffP = high[i] - prevHigh;
            prevHigh = high[i];
            float diffM = prevLow - low[i];
            prevLow = low[i];
            if (diffM > 0.0f && diffM > diffP)
              prevDM = prevDM - prevDM / period + diffM;
            else
              prevDM = prevDM - prevDM / period;
            out[i] = prevDM;
          }
          return out;
        }

        std::vector<float> plus_di_ref(
            const std::vector<float> &high, const std::vector<float> &low,
            const std::vector<float> &close, int period) {
          int N = high.size();
          std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
          float prevHigh = high[0];
          float prevLow = low[0];
          float prevClose = close[0];
          float dmp_s = 0.0f, tr_s = 0.0f;
          for (int i = 1; i < N; ++i) {
            float upMove = high[i] - prevHigh;
            float downMove = prevLow - low[i];
            float dmPlus = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
            float tr = std::max(high[i] - low[i],
                                std::max(std::fabs(high[i] - prevClose),
                                         std::fabs(low[i] - prevClose)));
            prevHigh = high[i];
            prevLow = low[i];
            prevClose = close[i];
            if (i <= period) {
              dmp_s += dmPlus;
              tr_s += tr;
              if (i == period)
                out[i] = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
            } else {
              dmp_s = dmp_s - dmp_s / period + dmPlus;
              tr_s = tr_s - tr_s / period + tr;
              out[i] = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
            }
          }
          return out;
        }

        std::vector<float> minus_di_ref(
            const std::vector<float> &high, const std::vector<float> &low,
            const std::vector<float> &close, int period) {
          int N = high.size();
          std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
          float prevHigh = high[0];
          float prevLow = low[0];
          float prevClose = close[0];
          float dmm_s = 0.0f, tr_s = 0.0f;
          for (int i = 1; i < N; ++i) {
            float upMove = high[i] - prevHigh;
            float downMove = prevLow - low[i];
            float dmMinus =
                (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
            float tr = std::max(high[i] - low[i],
                                std::max(std::fabs(high[i] - prevClose),
                                         std::fabs(low[i] - prevClose)));
            prevHigh = high[i];
            prevLow = low[i];
            prevClose = close[i];
            if (i <= period) {
              dmm_s += dmMinus;
              tr_s += tr;
              if (i == period)
                out[i] = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
            } else {
              dmm_s = dmm_s - dmm_s / period + dmMinus;
              tr_s = tr_s - tr_s / period + tr;
              out[i] = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
            }
          }
          return out;
        }
