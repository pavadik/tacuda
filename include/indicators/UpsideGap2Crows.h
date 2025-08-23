#ifndef UPSIDEGAP2CROWS_H
#define UPSIDEGAP2CROWS_H

#include "Indicator.h"

class UpsideGap2Crows : public Indicator {
public:
  UpsideGap2Crows() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif

