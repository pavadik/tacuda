#ifndef IDENTICALTHREECROWS_H
#define IDENTICALTHREECROWS_H

#include "Indicator.h"

class IdenticalThreeCrows : public Indicator {
public:
  IdenticalThreeCrows() = default;
  void calculate(const float *open, const float *high, const float *low,
                 const float *close, float *output, int size) noexcept(false);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;
};

#endif
