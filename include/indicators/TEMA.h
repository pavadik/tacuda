#ifndef TEMA_H
#define TEMA_H

#include "Indicator.h"

namespace tacuda {
class TEMA : public Indicator {
public:
    explicit TEMA(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
