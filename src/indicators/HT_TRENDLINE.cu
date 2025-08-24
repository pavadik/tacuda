#include <indicators/HT_TRENDLINE.h>
#include <utils/CudaUtils.h>
#include <cuda_runtime.h>
#include <limits>
#include <cmath>
#include <vector>

// Port of TA-Lib's HT_TRENDLINE implemented in C++.
static void ht_trendline_cpu(const std::vector<float>& in, std::vector<float>& out) {
    const int size = static_cast<int>(in.size());
    const int lookbackTotal = 63;
    if (size <= lookbackTotal)
        return;

    const double a = 0.0962;
    const double b = 0.5769;
    const double rad2Deg = 180.0 / (4.0 * std::atan(1.0));

    double period = 0.0, smoothPeriod = 0.0;
    double periodWMASub, periodWMASum, trailingWMAValue, smoothedValue;
    double hilbertTempReal;
    int trailingWMAIdx, hilbertIdx;
    double Q2 = 0.0, I2 = 0.0, prevQ2 = 0.0, prevI2 = 0.0, Re = 0.0, Im = 0.0;
    double I1ForOddPrev2 = 0.0, I1ForOddPrev3 = 0.0;
    double I1ForEvenPrev2 = 0.0, I1ForEvenPrev3 = 0.0;

    double detrender_Odd[3] = {0.0}, detrender_Even[3] = {0.0};
    double prev_detrender_Odd = 0.0, prev_detrender_Even = 0.0;
    double prev_detrender_input_Odd = 0.0, prev_detrender_input_Even = 0.0;
    double detrender = 0.0;

    double Q1_Odd[3] = {0.0}, Q1_Even[3] = {0.0};
    double prev_Q1_Odd = 0.0, prev_Q1_Even = 0.0;
    double prev_Q1_input_Odd = 0.0, prev_Q1_input_Even = 0.0;
    double Q1 = 0.0;

    double jI_Odd[3] = {0.0}, jI_Even[3] = {0.0};
    double prev_jI_Odd = 0.0, prev_jI_Even = 0.0;
    double prev_jI_input_Odd = 0.0, prev_jI_input_Even = 0.0;
    double jI = 0.0;

    double jQ_Odd[3] = {0.0}, jQ_Even[3] = {0.0};
    double prev_jQ_Odd = 0.0, prev_jQ_Even = 0.0;
    double prev_jQ_input_Odd = 0.0, prev_jQ_input_Even = 0.0;
    double jQ = 0.0;

    double iTrend1 = 0.0, iTrend2 = 0.0, iTrend3 = 0.0;

    int startIdx = lookbackTotal;
    int endIdx = size - 1;
    trailingWMAIdx = startIdx - lookbackTotal;
    int today = trailingWMAIdx;

    double tempReal = in[today++];
    periodWMASub = tempReal;
    periodWMASum = tempReal;
    tempReal = in[today++];
    periodWMASub += tempReal;
    periodWMASum += tempReal * 2.0;
    tempReal = in[today++];
    periodWMASub += tempReal;
    periodWMASum += tempReal * 3.0;
    trailingWMAValue = 0.0;

    auto doPriceWMA = [&](double newPrice, double& smoothed) {
        periodWMASub += newPrice;
        periodWMASub -= trailingWMAValue;
        periodWMASum += newPrice * 4.0;
        trailingWMAValue = in[trailingWMAIdx++];
        smoothed = periodWMASum * 0.1;
        periodWMASum -= periodWMASub;
    };

    for (int i = 0; i < 34; ++i) {
        tempReal = in[today++];
        doPriceWMA(tempReal, smoothedValue);
    }

    hilbertIdx = 0;

    while (today <= endIdx) {
        double adjustedPrevPeriod = (0.075 * period) + 0.54;
        double todayValue = in[today];
        doPriceWMA(todayValue, smoothedValue);

        if ((today & 1) == 0) {
            hilbertTempReal = a * smoothedValue;
            detrender = -detrender_Even[hilbertIdx];
            detrender_Even[hilbertIdx] = hilbertTempReal;
            detrender += hilbertTempReal;
            detrender -= prev_detrender_Even;
            prev_detrender_Even = b * prev_detrender_input_Even;
            detrender += prev_detrender_Even;
            prev_detrender_input_Even = smoothedValue;
            detrender *= adjustedPrevPeriod;

            hilbertTempReal = a * detrender;
            Q1 = -Q1_Even[hilbertIdx];
            Q1_Even[hilbertIdx] = hilbertTempReal;
            Q1 += hilbertTempReal;
            Q1 -= prev_Q1_Even;
            prev_Q1_Even = b * prev_Q1_input_Even;
            Q1 += prev_Q1_Even;
            prev_Q1_input_Even = detrender;
            Q1 *= adjustedPrevPeriod;

            hilbertTempReal = a * I1ForEvenPrev3;
            jI = -jI_Even[hilbertIdx];
            jI_Even[hilbertIdx] = hilbertTempReal;
            jI += hilbertTempReal;
            jI -= prev_jI_Even;
            prev_jI_Even = b * prev_jI_input_Even;
            jI += prev_jI_Even;
            prev_jI_input_Even = I1ForEvenPrev3;
            jI *= adjustedPrevPeriod;

            hilbertTempReal = a * Q1;
            jQ = -jQ_Even[hilbertIdx];
            jQ_Even[hilbertIdx] = hilbertTempReal;
            jQ += hilbertTempReal;
            jQ -= prev_jQ_Even;
            prev_jQ_Even = b * prev_jQ_input_Even;
            jQ += prev_jQ_Even;
            prev_jQ_input_Even = Q1;
            jQ *= adjustedPrevPeriod;

            if (++hilbertIdx == 3)
                hilbertIdx = 0;

            Q2 = (0.2 * (Q1 + jI)) + (0.8 * prevQ2);
            I2 = (0.2 * (I1ForEvenPrev3 - jQ)) + (0.8 * prevI2);

            I1ForOddPrev3 = I1ForOddPrev2;
            I1ForOddPrev2 = detrender;
        } else {
            hilbertTempReal = a * smoothedValue;
            detrender = -detrender_Odd[hilbertIdx];
            detrender_Odd[hilbertIdx] = hilbertTempReal;
            detrender += hilbertTempReal;
            detrender -= prev_detrender_Odd;
            prev_detrender_Odd = b * prev_detrender_input_Odd;
            detrender += prev_detrender_Odd;
            prev_detrender_input_Odd = smoothedValue;
            detrender *= adjustedPrevPeriod;

            hilbertTempReal = a * detrender;
            Q1 = -Q1_Odd[hilbertIdx];
            Q1_Odd[hilbertIdx] = hilbertTempReal;
            Q1 += hilbertTempReal;
            Q1 -= prev_Q1_Odd;
            prev_Q1_Odd = b * prev_Q1_input_Odd;
            Q1 += prev_Q1_Odd;
            prev_Q1_input_Odd = detrender;
            Q1 *= adjustedPrevPeriod;

            hilbertTempReal = a * I1ForOddPrev3;
            jI = -jI_Odd[hilbertIdx];
            jI_Odd[hilbertIdx] = hilbertTempReal;
            jI += hilbertTempReal;
            jI -= prev_jI_Odd;
            prev_jI_Odd = b * prev_jI_input_Odd;
            jI += prev_jI_Odd;
            prev_jI_input_Odd = I1ForOddPrev3;
            jI *= adjustedPrevPeriod;

            hilbertTempReal = a * Q1;
            jQ = -jQ_Odd[hilbertIdx];
            jQ_Odd[hilbertIdx] = hilbertTempReal;
            jQ += hilbertTempReal;
            jQ -= prev_jQ_Odd;
            prev_jQ_Odd = b * prev_jQ_input_Odd;
            jQ += prev_jQ_Odd;
            prev_jQ_input_Odd = Q1;
            jQ *= adjustedPrevPeriod;

            Q2 = (0.2 * (Q1 + jI)) + (0.8 * prevQ2);
            I2 = (0.2 * (I1ForOddPrev3 - jQ)) + (0.8 * prevI2);

            if (++hilbertIdx == 3)
                hilbertIdx = 0;

            I1ForEvenPrev3 = I1ForEvenPrev2;
            I1ForEvenPrev2 = detrender;
        }

        Re = (0.2 * ((I2 * prevI2) + (Q2 * prevQ2))) + (0.8 * Re);
        Im = (0.2 * ((I2 * prevQ2) - (Q2 * prevI2))) + (0.8 * Im);
        prevQ2 = Q2;
        prevI2 = I2;
        double temp = period;
        if ((Im != 0.0) && (Re != 0.0))
            period = 360.0 / (std::atan(Im / Re) * rad2Deg);
        double temp2 = 1.5 * temp;
        if (period > temp2) period = temp2;
        temp2 = 0.67 * temp;
        if (period < temp2) period = temp2;
        if (period < 6) period = 6;
        else if (period > 50) period = 50;
        period = (0.2 * period) + (0.8 * temp);
        smoothPeriod = (0.33 * period) + (0.67 * smoothPeriod);

        double DCPeriod = smoothPeriod + 0.5;
        int DCPeriodInt = static_cast<int>(DCPeriod);
        double avgPrice = 0.0;
        int idx = today;
        for (int i = 0; i < DCPeriodInt; ++i)
            avgPrice += in[idx--];
        if (DCPeriodInt > 0)
            avgPrice /= DCPeriodInt;

        double trend = (4.0 * avgPrice + 3.0 * iTrend1 + 2.0 * iTrend2 + iTrend3) / 10.0;
        iTrend3 = iTrend2;
        iTrend2 = iTrend1;
        iTrend1 = avgPrice;

        if (today >= startIdx)
            out[today] = static_cast<float>(trend);

        today++;
    }
}

void HT_TRENDLINE::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    std::vector<float> h_in(size);
    CUDA_CHECK(cudaMemcpy(h_in.data(), input, size * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> h_out(size, std::numeric_limits<float>::quiet_NaN());
    ht_trendline_cpu(h_in, h_out);
    CUDA_CHECK(cudaMemcpy(output, h_out.data(), size * sizeof(float), cudaMemcpyHostToDevice));
}

