#ifndef T3_H
#define T3_H

#include "Indicator.h"

class T3 : public Indicator {
public:
  T3(int period, float vFactor);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int period;
  float vFactor;
};

#endif
