#ifndef CANDLEUTILS_H
#define CANDLEUTILS_H

#include <cmath>

__host__ __device__ inline float real_body(float open, float close) {
  return fabsf(close - open);
}

__host__ __device__ inline float upper_shadow(float high, float open,
                                              float close) {
  return high - fmaxf(open, close);
}

__host__ __device__ inline float lower_shadow(float low, float open,
                                              float close) {
  return fminf(open, close) - low;
}

__host__ __device__ inline bool is_doji(float open, float high, float low,
                                        float close, float threshold = 0.1f) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  return real_body(open, close) / range <= threshold;
}

__host__ __device__ inline bool is_hammer(float open, float high, float low,
                                          float close) {
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return lower >= 2.0f * body && upper <= body;
}

__host__ __device__ inline bool is_inverted_hammer(float open, float high,
                                                   float low, float close) {
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return upper >= 2.0f * body && lower <= body;
}

__host__ __device__ inline bool
is_bullish_engulfing(float prevOpen, float prevClose, float open, float close) {
  return close > open && prevClose < prevOpen && open <= prevClose &&
         close >= prevOpen;
}

__host__ __device__ inline bool
is_bearish_engulfing(float prevOpen, float prevClose, float open, float close) {
  return close < open && prevClose > prevOpen && open >= prevClose &&
         close <= prevOpen;
}

__host__ __device__ inline bool
is_three_white_soldiers(float o1, float h1, float l1, float c1, float o2,
                        float h2, float l2, float c2, float o3, float h3,
                        float l3, float c3) {
  return c1 > o1 && c2 > o2 && c3 > o3 && o2 >= o1 && o2 <= c1 && o3 >= o2 &&
         o3 <= c2 && c1 < c2 && c2 < c3;
}

__host__ __device__ inline bool
is_abandoned_baby(float o1, float h1, float l1, float c1, float o2, float h2,
                  float l2, float c2, float o3, float h3, float l3, float c3) {
  bool bearish1 = c1 < o1;
  bool doji2 = is_doji(o2, h2, l2, c2);
  bool bullish3 = c3 > o3;
  bool gap1 = h2 < l1;
  bool gap2 = l3 > h2;
  return bearish1 && doji2 && bullish3 && gap1 && gap2;
}

__host__ __device__ inline bool is_advance_block(float o1, float h1, float l1,
                                                 float c1, float o2, float h2,
                                                 float l2, float c2, float o3,
                                                 float h3, float l3, float c3) {
  if (!(c1 > o1 && c2 > o2 && c3 > o3))
    return false;
  if (!(o2 >= o1 && o2 <= c1 && o3 >= o2 && o3 <= c2))
    return false;
  float b1 = c1 - o1;
  float b2 = c2 - o2;
  float b3 = c3 - o3;
  if (!(b1 > b2 && b2 > b3))
    return false;
  return true;
}

__host__ __device__ inline bool is_belt_hold(float open, float high, float low,
                                             float close) {
  float body = fabsf(close - open);
  float range = high - low;
  return open == low && close > open && body > range * 0.5f;
}

__host__ __device__ inline bool
is_breakaway(float o1, float h1, float l1, float c1, float o2, float h2,
             float l2, float c2, float o3, float h3, float l3, float c3,
             float o4, float h4, float l4, float c4, float o5, float h5,
             float l5, float c5) {
  if (!(c1 < o1 && c2 < o2 && c3 < o3 && c4 < o4 && c5 > o5))
    return false;
  if (!(o2 < c1 && o3 < c2 && o4 < c3))
    return false;
  if (!(c5 > o1))
    return false;
  return true;
}

__host__ __device__ inline bool is_two_crows(float o1, float h1, float l1,
                                             float c1, float o2, float h2,
                                             float l2, float c2, float o3,
                                             float h3, float l3, float c3) {
  return c1 > o1 && c2 < o2 && c3 < o3 && o2 > c1 && c2 > c1 && o3 < o2 &&
         o3 > c2 && c3 < c2 && c3 > o1 && c3 < c1;
}

__host__ __device__ inline bool
is_three_black_crows(float o1, float h1, float l1, float c1, float o2, float h2,
                     float l2, float c2, float o3, float h3, float l3,
                     float c3) {
  return c1 < o1 && c2 < o2 && c3 < o3 && o2 <= o1 && o2 >= c1 && o3 <= o2 &&
         o3 >= c2 && c1 > c2 && c2 > c3;
}

__host__ __device__ inline bool is_three_inside(float o1, float h1, float l1,
                                                float c1, float o2, float h2,
                                                float l2, float c2, float o3,
                                                float h3, float l3, float c3) {
  return c1 > o1 && c2 < o2 && o2 >= o1 && c2 <= c1 && o2 <= c1 && c2 >= o1 &&
         c3 > o3 && c3 > c1;
}

__host__ __device__ inline bool
is_three_line_strike(float o1, float h1, float l1, float c1, float o2, float h2,
                     float l2, float c2, float o3, float h3, float l3, float c3,
                     float o4, float h4, float l4, float c4) {
  bool firstThree = c1 > o1 && c2 > o2 && c3 > o3 && o2 >= o1 && o2 <= c1 &&
                    o3 >= o2 && o3 <= c2 && c1 < c2 && c2 < c3;
  bool fourth = c4 < o4 && o4 >= c3 && c4 < o1;
  return firstThree && fourth;
}

__host__ __device__ inline bool
is_three_stars_in_south(float o1, float h1, float l1, float c1, float o2,
                        float h2, float l2, float c2, float o3, float h3,
                        float l3, float c3) {
  float ls1 = lower_shadow(l1, o1, c1);
  float ls2 = lower_shadow(l2, o2, c2);
  float ls3 = lower_shadow(l3, o3, c3);
  return c1 < o1 && c2 < o2 && c3 < o3 && c1 > c2 && c2 > c3 && ls1 > ls2 &&
         ls2 > ls3;
}

__host__ __device__ inline bool is_closing_marubozu(float open, float high,
                                                    float low, float close) {
  return (close == high && open > low) || (close == low && open < high);
}

__host__ __device__ inline bool
is_conceal_baby_swallow(float o1, float h1, float l1, float c1, float o2,
                        float h2, float l2, float c2, float o3, float h3,
                        float l3, float c3, float o4, float h4, float l4,
                        float c4) {
  return c1 < o1 && c2 < o2 && c3 < o3 && c4 < o4 && c1 > c2 && c2 > c3 &&
         c3 > c4;
}

__host__ __device__ inline bool is_counterattack(float o1, float h1, float l1,
                                                 float c1, float o2, float h2,
                                                 float l2, float c2) {
  bool bullBear = c1 > o1 && c2 < o2;
  bool bearBull = c1 < o1 && c2 > o2;
  return (bullBear || bearBull) && fabsf(c2 - c1) <= 1e-3f;
}

__host__ __device__ inline bool is_dark_cloud_cover(float o1, float h1,
                                                    float l1, float c1,
                                                    float o2, float h2,
                                                    float l2, float c2) {
  float mid1 = (o1 + c1) * 0.5f;
  return c1 > o1 && o2 > h1 && c2 < o2 && c2 > o1 && c2 < mid1;
}

__host__ __device__ inline bool is_doji_star(float o1, float h1, float l1,
                                             float c1, float o2, float h2,
                                             float l2, float c2) {
  bool gapUp = l2 > h1;
  bool gapDown = h2 < l1;
  return !is_doji(o1, h1, l1, c1) && is_doji(o2, h2, l2, c2) &&
         (gapUp || gapDown);
}

__host__ __device__ inline bool is_dragonfly_doji(float open, float high,
                                                  float low, float close) {
  if (!is_doji(open, high, low, close))
    return false;
  float range = high - low;
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return upper <= range * 0.1f && lower >= range * 0.5f;
}

__host__ __device__ inline float engulfing(float prevOpen, float prevClose,
                                           float open, float close) {
  if (is_bullish_engulfing(prevOpen, prevClose, open, close))
    return 1.0f;
  if (is_bearish_engulfing(prevOpen, prevClose, open, close))
    return -1.0f;
  return 0.0f;
}

__host__ __device__ inline bool is_evening_star(float o1, float h1, float l1,
                                                float c1, float o2, float h2,
                                                float l2, float c2, float o3,
                                                float h3, float l3, float c3) {
  bool bullish1 = c1 > o1;
  bool gapUp = l2 > h1;
  float body1 = real_body(o1, c1);
  float body2 = real_body(o2, c2);
  bool smallBody2 = body2 < body1 * 0.5f;
  bool bearish3 = c3 < o3;
  bool closeIntoBody = c3 < (o1 + c1) * 0.5f;
  return bullish1 && gapUp && smallBody2 && bearish3 && closeIntoBody;
}

__host__ __device__ inline bool
is_evening_doji_star(float o1, float h1, float l1, float c1, float o2, float h2,
                     float l2, float c2, float o3, float h3, float l3,
                     float c3) {
  bool bullish1 = c1 > o1;
  bool doji2 = is_doji(o2, h2, l2, c2);
  bool gapUp = l2 > h1;
  bool bearish3 = c3 < o3;
  bool closeIntoBody = c3 < (o1 + c1) * 0.5f;
  return bullish1 && doji2 && gapUp && bearish3 && closeIntoBody;
}

__host__ __device__ inline bool is_gravestone_doji(float open, float high,
                                                   float low, float close) {
  if (!is_doji(open, high, low, close))
    return false;
  float range = high - low;
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return upper >= range * 0.5f && lower <= range * 0.1f;
}

__host__ __device__ inline bool is_hanging_man(float open, float high,
                                               float low, float close) {
  return is_hammer(open, high, low, close);
}

__host__ __device__ inline bool
is_gap_side_side_white(float o1, float h1, float l1, float c1, float o2,
                       float h2, float l2, float c2, float o3, float h3,
                       float l3, float c3) {
  return c1 > o1 && c2 > o2 && c3 > o3 && l2 > h1 && l3 > h1 &&
         fabsf(o2 - o3) <= 1e-3f;
}

__host__ __device__ inline float harami(float prevOpen, float prevClose,
                                        float open, float close) {
  bool bullish = prevClose < prevOpen && open > prevClose && close < prevOpen &&
                 close > open;
  bool bearish = prevClose > prevOpen && open < prevClose && close > prevOpen &&
                 close < open;
  if (bullish)
    return 1.0f;
  if (bearish)
    return -1.0f;
  return 0.0f;
}

__host__ __device__ inline float harami_cross(float o1, float h1, float l1,
                                              float c1, float o2, float h2,
                                              float l2, float c2) {
  bool doji2 = is_doji(o2, h2, l2, c2);
  bool bullish = c1 < o1 && doji2 && o2 > c1 && c2 < o1;
  bool bearish = c1 > o1 && doji2 && o2 < c1 && c2 > o1;
  if (bullish)
    return 1.0f;
  if (bearish)
    return -1.0f;
  return 0.0f;
}

__host__ __device__ inline bool is_high_wave(float open, float high, float low,
                                             float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return body <= range * 0.3f && upper >= range * 0.4f && lower >= range * 0.4f;
}

__host__ __device__ inline float hikkake(float o1, float h1, float l1, float c1,
                                         float o2, float h2, float l2, float c2,
                                         float o3, float h3, float l3,
                                         float c3) {
  bool inside = h2 <= h1 && l2 >= l1;
  bool bullish = inside && h3 > h2 && l3 > l2 && c3 > o3;
  bool bearish = inside && h3 < h2 && l3 < l2 && c3 < o3;
  if (bullish)
    return 1.0f;
  if (bearish)
    return -1.0f;
  return 0.0f;
}

__host__ __device__ inline float
hikkake_mod(float o1, float h1, float l1, float c1, float o2, float h2,
            float l2, float c2, float o3, float h3, float l3, float c3,
            float o4, float h4, float l4, float c4) {
  bool inside = h2 <= h1 && l2 >= l1;
  bool bullBreak = inside && h3 > h2 && l3 > l2 && c3 > o3 && c4 > h2;
  bool bearBreak = inside && h3 < h2 && l3 < l2 && c3 < o3 && c4 < l2;
  if (bullBreak)
    return 1.0f;
  if (bearBreak)
    return -1.0f;
  return 0.0f;
}

__host__ __device__ inline bool is_homing_pigeon(float o1, float h1, float l1,
                                                 float c1, float o2, float h2,
                                                 float l2, float c2) {
  return c1 < o1 && c2 < o2 && o2 >= c1 && o2 <= o1 && c2 >= c1 && c2 <= o1;
}

__host__ __device__ inline bool
is_identical_three_crows(float o1, float h1, float l1, float c1, float o2,
                         float h2, float l2, float c2, float o3, float h3,
                         float l3, float c3) {
  return c1 < o1 && c2 < o2 && c3 < o3 && o2 <= o1 && o2 >= c1 && o3 <= o2 &&
         o3 >= c2 && c1 > c2 && fabsf(c2 - c3) <= 1e-6f;
}

__host__ __device__ inline bool is_in_neck(float o1, float h1, float l1,
                                           float c1, float o2, float h2,
                                           float l2, float c2) {
  float body1 = real_body(o1, c1);
  return c1 < o1 && c2 > o2 && o2 < l1 && c2 >= c1 && c2 <= c1 + body1 * 0.1f;
}

__host__ __device__ inline bool is_marubozu(float open, float high, float low,
                                            float close) {
  return upper_shadow(high, open, close) <= 1e-6f &&
         lower_shadow(low, open, close) <= 1e-6f;
}

__host__ __device__ inline bool is_matching_low(float o1, float h1, float l1,
                                                float c1, float o2, float h2,
                                                float l2, float c2) {
  return c1 < o1 && c2 < o2 && fabsf(c2 - c1) <= 1e-6f;
}

__host__ __device__ inline bool
is_ladder_bottom(float o1, float h1, float l1, float c1, float o2, float h2,
                 float l2, float c2, float o3, float h3, float l3, float c3,
                 float o4, float h4, float l4, float c4, float o5, float h5,
                 float l5, float c5) {
  return c1 < o1 && c2 < o2 && c3 < o3 && c1 > c2 && c2 > c3 && c4 < o4 &&
         c4 > c3 && c5 > o5 && c5 > o4 && c5 > c4;
}

__host__ __device__ inline bool is_long_legged_doji(float open, float high,
                                                    float low, float close) {
  if (!is_doji(open, high, low, close))
    return false;
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return upper >= range * 0.4f && lower >= range * 0.4f;
}

__host__ __device__ inline bool is_long_line(float open, float high, float low,
                                             float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return body >= range * 0.7f && upper <= range * 0.15f &&
         lower <= range * 0.15f;
}

__host__ __device__ inline bool is_kicking(float o1, float h1, float l1,
                                           float c1, float o2, float h2,
                                           float l2, float c2) {
  bool firstBull = c1 > o1;
  bool firstBear = c1 < o1;
  bool secondBull = c2 > o2;
  bool secondBear = c2 < o2;
  bool opposite = (firstBull && secondBear) || (firstBear && secondBull);
  if (!opposite)
    return false;
  if (!is_marubozu(o1, h1, l1, c1) || !is_marubozu(o2, h2, l2, c2))
    return false;
  if (firstBull)
    return o2 < l1;
  else
    return o2 > h1;
}

__host__ __device__ inline bool is_kicking_by_length(float o1, float h1,
                                                     float l1, float c1,
                                                     float o2, float h2,
                                                     float l2, float c2) {
  return is_kicking(o1, h1, l1, c1, o2, h2, l2, c2) &&
         real_body(o2, c2) > real_body(o1, c1);
}

__host__ __device__ inline bool is_short_line(float open, float high, float low,
                                              float close) {
  float range = high - low;
  if (range <= 0.0f)
    return false;
  float body = real_body(open, close);
  float upper = upper_shadow(high, open, close);
  float lower = lower_shadow(low, open, close);
  return body <= range * 0.3f && upper <= range * 0.3f && lower <= range * 0.3f;
}

__host__ __device__ inline bool is_shooting_star(float open, float high,
                                                 float low, float close) {
  return is_inverted_hammer(open, high, low, close);
}

__host__ __device__ inline bool is_rickshaw_man(float open, float high,
                                                float low, float close) {
  if (!is_long_legged_doji(open, high, low, close))
    return false;
  float mid = (high + low) * 0.5f;
  float bodyMid = (open + close) * 0.5f;
  return fabsf(bodyMid - mid) <= (high - low) * 0.1f;
}

__host__ __device__ inline float separating_lines(float o1, float h1, float l1,
                                                  float c1, float o2, float h2,
                                                  float l2, float c2) {
  bool bullish = c1 < o1 && c2 > o2 && fabsf(o2 - o1) <= 1e-3f && c2 > c1;
  bool bearish = c1 > o1 && c2 < o2 && fabsf(o2 - o1) <= 1e-3f && c2 < c1;
  if (bullish)
    return 1.0f;
  if (bearish)
    return -1.0f;
  return 0.0f;
}

__host__ __device__ inline float
rise_fall_3_methods(float o1, float h1, float l1, float c1, float o2, float h2,
                    float l2, float c2, float o3, float h3, float l3, float c3,
                    float o4, float h4, float l4, float c4, float o5, float h5,
                    float l5, float c5) {
  bool rising = c1 > o1 && c2 < o2 && c3 < o3 && c4 < o4 && h2 <= h1 &&
                l2 >= l1 && h3 <= h1 && l3 >= l1 && h4 <= h1 && l4 >= l1 &&
                c5 > o5 && c5 > c1;
  bool falling = c1 < o1 && c2 > o2 && c3 > o3 && c4 > o4 && h2 <= h1 &&
                 l2 >= l1 && h3 <= h1 && l3 >= l1 && h4 <= h1 && l4 >= l1 &&
                 c5 < o5 && c5 < c1;
  if (rising)
    return 1.0f;
  if (falling)
    return -1.0f;
  return 0.0f;
}

#endif
