#ifndef SHOOTINGSTAR_H
#define SHOOTINGSTAR_H

#include "Indicator.h"

class ShootingStar : public Indicator {
public:
  ShootingStar() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
