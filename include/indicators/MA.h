#ifndef MA_H
#define MA_H

#include "Indicator.h"

namespace tacuda {
enum class MAType {
    SMA = 0,
    EMA = 1,
    WMA = 2,
    DEMA = 3,
    TEMA = 4,
    TRIMA = 5,
    KAMA = 6,
    MAMA = 7,
    T3 = 8,
};

class MA : public Indicator {
public:
    MA(int period, MAType type);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
    MAType type;
};

} // namespace tacuda

#endif
