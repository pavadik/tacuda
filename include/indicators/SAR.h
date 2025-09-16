#ifndef SAR_H
#define SAR_H

#include "Indicator.h"

namespace tacuda {
class SAR : public Indicator {
public:
    explicit SAR(float step, float maxAcceleration);
    void calculate(const float* high, const float* low,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    float step;
    float maxAcceleration;
};

} // namespace tacuda

#endif
