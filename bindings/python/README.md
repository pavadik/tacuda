# Python bindings

Usage:
```python
import numpy as np
from tacuda import sma, momentum, macd

x = np.random.rand(1024).astype(np.float32)
print(sma(x, 14)[:10])
print(momentum(x, 10)[:10])
line, signal, hist = macd(x)
print(line[:10])
print(signal[:10])
print(hist[:10])
```
Make sure you've built the shared library (`tacuda`) so the bindings can locate it.
