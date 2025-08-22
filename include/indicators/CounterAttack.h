#ifndef COUNTERATTACK_H
#define COUNTERATTACK_H

#include "Indicator.h"

class CounterAttack : public Indicator {
public:
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
