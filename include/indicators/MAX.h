#ifndef MAX_H
#define MAX_H

#include "Indicator.h"

class MAX : public Indicator {
public:
    explicit MAX(int period);
    void calculate(const float* input, float* output, int size) noexcept(false) override;
private:
    int period;
};

#endif
