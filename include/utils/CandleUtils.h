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

#endif
