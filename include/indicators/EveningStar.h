#ifndef EVENINGSTAR_H
#define EVENINGSTAR_H

#include "Indicator.h"

class EveningStar : public Indicator {
public:
  EveningStar() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
