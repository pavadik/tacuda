#ifndef DOJI_H
#define DOJI_H

#include "Indicator.h"

namespace tacuda {
class Doji : public Indicator {
public:
    explicit Doji(float threshold = 0.1f);
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    float threshold;
};

} // namespace tacuda

#endif
