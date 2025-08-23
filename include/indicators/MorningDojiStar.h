#ifndef MORNINGDOJISTAR_H
#define MORNINGDOJISTAR_H

#include "Indicator.h"

class MorningDojiStar : public Indicator {
public:
  MorningDojiStar() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
