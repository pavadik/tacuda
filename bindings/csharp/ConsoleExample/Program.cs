using System;
using Tacuda.Bindings;

int length = 1024;
var input = new float[length];
for (int i = 0; i < input.Length; ++i)
{
    input[i] = (float)Math.Sin(0.01 * i);
}

var output = new float[length];
int status = NativeMethods.ct_sma(input, output, length, period: 14);
if (status != 0)
{
    throw new InvalidOperationException($"ct_sma failed with status {status}");
}

Console.WriteLine($"SMA[0..4]: {string.Join(", ", output[0], output[1], output[2], output[3], output[4])}");
