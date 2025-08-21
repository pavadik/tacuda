"""
Python bindings for cuda_talib via ctypes.
Build the project first (shared library 'cuda_talib_shared').
"""
import ctypes
import os
import sys

def _load_lib():
    names = []
    if sys.platform.startswith("win"):
        names = ["cuda_talib_shared.dll"]
    elif sys.platform == "darwin":
        names = ["libcuda_talib_shared.dylib", "cuda_talib_shared.dylib"]
    else:
        names = ["libcuda_talib_shared.so", "cuda_talib_shared.so"]
    search_paths = [os.getcwd(), os.path.join(os.getcwd(), "build"), os.path.dirname(__file__),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))]
    for base in search_paths:
        for n in names:
            p = os.path.join(base, n)
            if os.path.exists(p):
                return ctypes.CDLL(p)
    raise OSError("cuda_talib_shared library not found. Build the project first.")

_lib = _load_lib()

_lib.ct_sma.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_sma.restype  = ctypes.c_int
_lib.ct_momentum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_momentum.restype  = ctypes.c_int
_lib.ct_macd_line.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ct_macd_line.restype  = ctypes.c_int

def _as_float_ptr(arr):
    import numpy as np
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def sma(x, period):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    xin, pin = _as_float_ptr(x)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_sma(pin, pout, x.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_sma failed")
    return out

def momentum(x, period):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    xin, pin = _as_float_ptr(x)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_momentum(pin, pout, x.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_momentum failed")
    return out

def macd_line(x, fast=12, slow=26, signal=9):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    xin, pin = _as_float_ptr(x)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_macd_line(pin, pout, x.size, int(fast), int(slow), int(signal))
    if rc != 0:
        raise RuntimeError("ct_macd_line failed")
    return out
