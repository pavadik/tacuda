#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include <gtest/gtest.h>
#include <cmath>

void expect_approx_equal(const std::vector<float>& a,
                         const std::vector<float>& b,
                         float eps = 1e-3f);

std::vector<float> dema_ref(const std::vector<float>& in, int period);
std::vector<float> tema_ref(const std::vector<float>& in, int period);
std::vector<float> trix_ref(const std::vector<float>& in, int period);
std::vector<float> kama_ref(const std::vector<float>& in, int period,
                            int fastPeriod, int slowPeriod);
std::vector<float> ht_dcperiod_ref(const std::vector<float>& in);
std::vector<float> ht_dcphase_ref(const std::vector<float>& in);
std::pair<std::vector<float>, std::vector<float>> ht_phasor_ref(const std::vector<float>& in);
std::pair<std::vector<float>, std::vector<float>> ht_sine_ref(const std::vector<float>& in);
std::vector<float> ht_trendmode_ref(const std::vector<float>& in);
std::vector<float> sar_ref(const std::vector<float>& high,
                           const std::vector<float>& low,
                           float step, float maxAcc);
std::vector<float> adosc_ref(const std::vector<float>& high,
                             const std::vector<float>& low,
                             const std::vector<float>& close,
                             const std::vector<float>& volume,
                             int shortP, int longP);
std::vector<float> ultosc_ref(const std::vector<float>& high,
                              const std::vector<float>& low,
                              const std::vector<float>& close,
                              int shortP, int medP, int longP);
std::vector<float> ad_ref(const std::vector<float>& high,
                          const std::vector<float>& low,
                          const std::vector<float>& close,
                          const std::vector<float>& volume);
std::vector<float> apo_ref(const std::vector<float>& in, int fastP, int slowP);
std::vector<float> aroonosc_ref(const std::vector<float>& high,
                                const std::vector<float>& low,
                                int period);
std::vector<float> adxr_ref(const std::vector<float>& high,
                            const std::vector<float>& low,
                            const std::vector<float>& close, int p);
std::vector<float> avgprice_ref(const std::vector<float>& open,
                                const std::vector<float>& high,
                                const std::vector<float>& low,
                                const std::vector<float>& close);
std::vector<float> linearreg_ref(const std::vector<float>& in, int period);
std::vector<float> linearreg_slope_ref(const std::vector<float>& in, int period);
std::vector<float> linearreg_intercept_ref(const std::vector<float>& in, int period);
std::vector<float> linearreg_angle_ref(const std::vector<float>& in, int period);
