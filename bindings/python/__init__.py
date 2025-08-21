"""
Python bindings for tacuda via ctypes.
Build the project first (shared library 'tacuda').
"""
import ctypes
import os
import sys

def _load_lib():
    names = []
    if sys.platform.startswith("win"):
        names = ["tacuda.dll"]
    elif sys.platform == "darwin":
        names = ["libtacuda.dylib", "tacuda.dylib"]
    else:
        names = ["libtacuda.so", "tacuda.so"]
    search_paths = [os.getcwd(), os.path.join(os.getcwd(), "build"), os.path.dirname(__file__),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))]
    for base in search_paths:
        for n in names:
            p = os.path.join(base, n)
            if os.path.exists(p):
                return ctypes.CDLL(p)
    raise OSError("tacuda library not found. Build the project first.")

_lib = _load_lib()

_lib.ct_sma.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_sma.restype  = ctypes.c_int
_lib.ct_momentum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_momentum.restype  = ctypes.c_int
_lib.ct_macd.argtypes = [ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.POINTER(ctypes.c_float),
                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ct_macd.restype  = ctypes.c_int

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

def macd(x, fast=12, slow=26, signal=9):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    line = np.zeros_like(x)
    sig = np.zeros_like(x)
    hist = np.zeros_like(x)
    xin, pin = _as_float_ptr(x)
    _, pline = _as_float_ptr(line)
    _, psig = _as_float_ptr(sig)
    _, phist = _as_float_ptr(hist)
    rc = _lib.ct_macd(pin, pline, psig, phist, x.size, int(fast), int(slow), int(signal))
    if rc != 0:
        raise RuntimeError("ct_macd failed")
    return line, sig, hist
