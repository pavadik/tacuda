#ifndef ACCBANDS_H
#define ACCBANDS_H

#include "Indicator.h"

namespace tacuda {
class ACCBANDS : public Indicator {
public:
    explicit ACCBANDS(int period);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
