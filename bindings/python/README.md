# Python bindings

Usage:
```python
import numpy as np
from tacuda import sma, momentum, macd_line

x = np.random.rand(1024).astype(np.float32)
print(sma(x, 14)[:10])
print(momentum(x, 10)[:10])
print(macd_line(x)[:10])
```
Make sure you've built the shared library (`tacuda`) so the bindings can locate it.
