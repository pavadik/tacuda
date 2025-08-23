using System;
using System.Runtime.InteropServices;

namespace CudaTaLib {
public static class Native {
#if WINDOWS
  const string LIB = "tacuda.dll";
#elif OSX
  const string LIB = "libtacuda.dylib";
#else
  const string LIB = "libtacuda.so";
#endif

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_sma(float[] input, float[] output, int size,
                                  int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_wma(float[] input, float[] output, int size,
                                  int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_momentum(float[] input, float[] output, int size,
                                       int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_macd_line(float[] input, float[] output, int size,
                                        int fast, int slow);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_rsi(float[] input, float[] output, int size,
                                  int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_atr(float[] high, float[] low, float[] close,
                                  float[] output, int size, int period,
                                  float initial);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_stochastic(float[] high, float[] low,
                                         float[] close, float[] kOut,
                                         float[] dOut, int size, int kPeriod,
                                         int dPeriod);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_cci(float[] high, float[] low, float[] close,
                                  float[] output, int size, int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_adx(float[] high, float[] low, float[] close,
                                  float[] output, int size, int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_ultosc(float[] high, float[] low, float[] close,
                                     float[] output, int size, int shortPeriod,
                                     int mediumPeriod, int longPeriod);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_sar(float[] high, float[] low, float[] output,
                                  int size, float step, float maxAcceleration);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_obv(float[] price, float[] volume, float[] output,
                                  int size);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_aroon(float[] high, float[] low, float[] up,
                                    float[] down, float[] osc, int size,
                                    int upPeriod, int downPeriod);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_trange(float[] high, float[] low, float[] close,
                                     float[] output, int size);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_sum(float[] input, float[] output, int size,
                                  int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_t3(float[] input, float[] output, int size,
                                 int period, float vFactor);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_trima(float[] input, float[] output, int size,
                                    int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_stochrsi(float[] input, float[] kOut,
                                       float[] dOut, int size, int rsiPeriod,
                                       int kPeriod, int dPeriod);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_rocp(float[] input, float[] output, int size,
                                   int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_rocr(float[] input, float[] output, int size,
                                   int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_minindex(float[] input, float[] output, int size,
                                       int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_minmax(float[] input, float[] minOut,
                                     float[] maxOut, int size, int period);

  [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
  public static extern int ct_minmaxindex(float[] input, float[] minIdx,
                                          float[] maxIdx, int size, int period);
}

public class Example {
  public static void Main(string[] args) {
    int N = 1024;
    var x = new float[N];
    for (int i = 0; i < N; ++i)
      x[i] = (float)Math.Sin(0.01 * i);

    var outArr = new float[N];
    int rc = Native.ct_sma(x, outArr, N, 14);
    if (rc != 0)
      throw new Exception("ct_sma failed");
            Console.WriteLine("SMA[0..4]: " + string.Join(\", \", outArr[0], outArr[1], outArr[2], outArr[3], outArr[4]));
  }
}
}
