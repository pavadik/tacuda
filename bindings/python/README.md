# Python bindings

The Python API is backed by static metadata generated from the public
`tacuda.h` header.  Callers interact with NumPy arrays while the runtime
handles conversion to `ctypes` pointers and output buffer allocation.

## Usage

```python
import numpy as np
from tacuda import sma, momentum, macd_line, rsi

x = np.random.rand(1024).astype(np.float32)
print(sma(x, 14)[:10])
print(momentum(x, 10)[:10])
print(macd_line(x)[:10])
print(rsi(x, 14)[:10])
```

Ensure the shared library (`tacuda`) is built and discoverable before
importing the module.

## Regenerating metadata

When the C header changes run the generator from the repository root:

```bash
python bindings/generate_bindings.py
```

The resulting `bindings/python/_generated.py` is tracked in the repository
and validated by CTest.
