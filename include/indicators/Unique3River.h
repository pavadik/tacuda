#ifndef UNIQUE3RIVER_H
#define UNIQUE3RIVER_H

#include "Indicator.h"

class Unique3River : public Indicator {
public:
  Unique3River() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif

