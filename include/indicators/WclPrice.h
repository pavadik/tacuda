#ifndef WCLPRICE_H
#define WCLPRICE_H

#include "Indicator.h"

class WclPrice : public Indicator {
public:
  WclPrice() = default;
  void calculate(const float *high, const float *low, const float *close,
                 float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
