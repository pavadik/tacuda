#ifndef CHANGE_H
#define CHANGE_H

#include "Indicator.h"

class Change : public Indicator {
public:
    explicit Change(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
