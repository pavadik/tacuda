#ifndef MINMAX_H
#define MINMAX_H

#include "Indicator.h"

class MINMAX : public Indicator {
public:
  explicit MINMAX(int period);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int period;
};

#endif
