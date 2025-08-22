#ifndef ENGULFING_H
#define ENGULFING_H

#include "Indicator.h"

class Engulfing : public Indicator {
public:
  Engulfing() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
