#ifndef INDICATOR_H
#define INDICATOR_H

class Indicator {
public:
    virtual void calculate(const float* input, float* output, int size) noexcept(false) = 0;
    virtual ~Indicator() = default;
};

#endif
