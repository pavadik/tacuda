#ifndef CANDLEUTILS_H
#define CANDLEUTILS_H

#include <cmath>

__host__ __device__ inline float real_body(float open, float close) {
    return fabsf(close - open);
}

__host__ __device__ inline float upper_shadow(float high, float open, float close) {
    return high - fmaxf(open, close);
}

__host__ __device__ inline float lower_shadow(float low, float open, float close) {
    return fminf(open, close) - low;
}

__host__ __device__ inline bool is_doji(float open, float high, float low, float close, float threshold = 0.1f) {
    float range = high - low;
    if (range <= 0.0f) return false;
    return real_body(open, close) / range <= threshold;
}

__host__ __device__ inline bool is_hammer(float open, float high, float low, float close) {
    float body = real_body(open, close);
    float upper = upper_shadow(high, open, close);
    float lower = lower_shadow(low, open, close);
    return lower >= 2.0f * body && upper <= body;
}

__host__ __device__ inline bool is_inverted_hammer(float open, float high, float low, float close) {
    float body = real_body(open, close);
    float upper = upper_shadow(high, open, close);
    float lower = lower_shadow(low, open, close);
    return upper >= 2.0f * body && lower <= body;
}

__host__ __device__ inline bool is_bullish_engulfing(float prevOpen, float prevClose, float open, float close) {
    return close > open && prevClose < prevOpen && open <= prevClose && close >= prevOpen;
}

__host__ __device__ inline bool is_bearish_engulfing(float prevOpen, float prevClose, float open, float close) {
    return close < open && prevClose > prevOpen && open >= prevClose && close <= prevOpen;
}

__host__ __device__ inline bool is_three_white_soldiers(
    float o1, float h1, float l1, float c1,
    float o2, float h2, float l2, float c2,
    float o3, float h3, float l3, float c3) {
    return c1 > o1 && c2 > o2 && c3 > o3 &&
           o2 >= o1 && o2 <= c1 &&
           o3 >= o2 && o3 <= c2 &&
           c1 < c2 && c2 < c3;
}

__host__ __device__ inline bool is_abandoned_baby(
    float o1, float h1, float l1, float c1,
    float o2, float h2, float l2, float c2,
    float o3, float h3, float l3, float c3) {
    bool bearish1 = c1 < o1;
    bool doji2 = is_doji(o2, h2, l2, c2);
    bool bullish3 = c3 > o3;
    bool gap1 = h2 < l1;
    bool gap2 = l3 > h2;
    return bearish1 && doji2 && bullish3 && gap1 && gap2;
}

__host__ __device__ inline bool is_advance_block(
    float o1, float h1, float l1, float c1,
    float o2, float h2, float l2, float c2,
    float o3, float h3, float l3, float c3) {
    if (!(c1 > o1 && c2 > o2 && c3 > o3)) return false;
    if (!(o2 >= o1 && o2 <= c1 && o3 >= o2 && o3 <= c2)) return false;
    float b1 = c1 - o1;
    float b2 = c2 - o2;
    float b3 = c3 - o3;
    if (!(b1 > b2 && b2 > b3)) return false;
    return true;
}

__host__ __device__ inline bool is_belt_hold(float open, float high, float low, float close) {
    float body = fabsf(close - open);
    float range = high - low;
    return open == low && close > open && body > range * 0.5f;
}

__host__ __device__ inline bool is_breakaway(
    float o1, float h1, float l1, float c1,
    float o2, float h2, float l2, float c2,
    float o3, float h3, float l3, float c3,
    float o4, float h4, float l4, float c4,
    float o5, float h5, float l5, float c5) {
    if (!(c1 < o1 && c2 < o2 && c3 < o3 && c4 < o4 && c5 > o5)) return false;
    if (!(o2 < c1 && o3 < c2 && o4 < c3)) return false;
    if (!(c5 > o1)) return false;
    return true;
}

#endif
