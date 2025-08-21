using System;
using System.Runtime.InteropServices;

namespace CudaTaLib
{
    public static class Native
    {
        #if WINDOWS
            const string LIB = "tacuda.dll";
        #elif OSX
            const string LIB = "libtacuda.dylib";
        #else
            const string LIB = "libtacuda.so";
        #endif

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ct_sma(float[] input, float[] output, int size, int period);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ct_momentum(float[] input, float[] output, int size, int period);

        [DllImport(LIB, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ct_macd_line(float[] input, float[] output, int size, int fast, int slow, int signal);
    }

    public class Example
    {
        public static void Main(string[] args)
        {
            int N = 1024;
            var x = new float[N];
            for (int i = 0; i < N; ++i) x[i] = (float)Math.Sin(0.01 * i);

            var outArr = new float[N];
            int rc = Native.ct_sma(x, outArr, N, 14);
            if (rc != 0) throw new Exception("ct_sma failed");
            Console.WriteLine("SMA[0..4]: " + string.Join(\", \", outArr[0], outArr[1], outArr[2], outArr[3], outArr[4]));
        }
    }
}
