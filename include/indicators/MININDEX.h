#ifndef MININDEX_H
#define MININDEX_H

#include "Indicator.h"

class MININDEX : public Indicator {
public:
  explicit MININDEX(int period);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int period;
};

#endif
