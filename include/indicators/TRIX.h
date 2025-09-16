#ifndef TRIX_H
#define TRIX_H

#include "Indicator.h"

namespace tacuda {
class TRIX : public Indicator {
public:
    explicit TRIX(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
