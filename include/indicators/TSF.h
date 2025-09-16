#ifndef TSF_H
#define TSF_H

#include "Indicator.h"

namespace tacuda {
class TSF : public Indicator {
public:
    explicit TSF(int period);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
