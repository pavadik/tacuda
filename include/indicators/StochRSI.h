#ifndef STOCHRSI_H
#define STOCHRSI_H

#include "Indicator.h"

class StochRSI : public Indicator {
public:
  StochRSI(int rsiPeriod, int kPeriod, int dPeriod);
  void calculate(const float *input, float *output,
                 int size) noexcept(false) override;

private:
  int rsiPeriod;
  int kPeriod;
  int dPeriod;
};

#endif
