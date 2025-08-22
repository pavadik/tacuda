#ifndef MA_H
#define MA_H

#include "Indicator.h"

enum class MAType { SMA = 0, EMA = 1 };

class MA : public Indicator {
public:
    MA(int period, MAType type);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
    MAType type;
};

#endif
