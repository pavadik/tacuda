"""
Python bindings for tacuda via ctypes.
Build the project first (shared library 'tacuda').
"""
import ctypes
import ctypes.util
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

    # Environment variables for explicit locations
    env_candidates = []
    env = os.environ.get("TACUDA_LIBRARY")
    if env:
        env_candidates.append(env)
    env_dir = os.environ.get("TACUDA_LIBRARY_PATH") or os.environ.get("TACUDA_LIBRARY_DIR")
    if env_dir:
        for n in names:
            env_candidates.append(os.path.join(env_dir, n))
    for candidate in env_candidates:
        if candidate and os.path.exists(candidate):
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                pass

    # Try system paths via ctypes.util.find_library
    libname = ctypes.util.find_library("tacuda")
    if libname:
        try:
            return ctypes.CDLL(libname)
        except OSError:
            pass

    # Fall back to package-relative locations
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    search_paths = [
        os.getcwd(),
        os.path.join(os.getcwd(), "build"),
        here,
        os.path.join(here, "lib"),
        root,
        os.path.join(root, "lib"),
        os.path.join(root, "build"),
    ]
    for base in search_paths:
        for n in names:
            p = os.path.join(base, n)
            if os.path.exists(p):
                try:
                    return ctypes.CDLL(p)
                except OSError:
                    continue
    raise OSError(
        "tacuda library not found. Build the project first or set TACUDA_LIBRARY[_PATH]."
    )

_lib = _load_lib()

_lib.ct_sma.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_sma.restype  = ctypes.c_int
_lib.ct_wma.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_wma.restype  = ctypes.c_int
_lib.ct_momentum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_momentum.restype  = ctypes.c_int
_lib.ct_macd_line.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                              ctypes.c_int, ctypes.c_int]
_lib.ct_macd_line.restype  = ctypes.c_int
_lib.ct_rsi.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_rsi.restype  = ctypes.c_int
_lib.ct_atr.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int, ctypes.c_int, ctypes.c_float]
_lib.ct_atr.restype  = ctypes.c_int
_lib.ct_stochastic.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ct_stochastic.restype  = ctypes.c_int
_lib.ct_cci.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int, ctypes.c_int]
_lib.ct_cci.restype  = ctypes.c_int
_lib.ct_adx.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int, ctypes.c_int]
_lib.ct_adx.restype  = ctypes.c_int
_lib.ct_obv.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_obv.restype  = ctypes.c_int

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

def wma(x, period):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    xin, pin = _as_float_ptr(x)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_wma(pin, pout, x.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_wma failed")
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

def macd_line(x, fast=12, slow=26):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    xin, pin = _as_float_ptr(x)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_macd_line(pin, pout, x.size, int(fast), int(slow))
    if rc != 0:
        raise RuntimeError("ct_macd_line failed")
    return out

def rsi(x, period):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    xin, pin = _as_float_ptr(x)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_rsi(pin, pout, x.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_rsi failed")
    return out

def atr(high, low, close, period, initial=0.0):
    import numpy as np
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if high.shape != low.shape or high.shape != close.shape:
        raise ValueError("high, low, close must have same shape")
    out = np.zeros_like(close)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_atr(ph, pl, pc, pout, close.size, int(period), float(initial))
    if rc != 0:
        raise RuntimeError("ct_atr failed")
    return out

def stochastic(high, low, close, k_period, d_period):
    import numpy as np
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if high.shape != low.shape or high.shape != close.shape:
        raise ValueError("high, low, close must have same shape")
    k = np.zeros_like(close)
    d = np.zeros_like(close)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pk = _as_float_ptr(k)
    _, pd = _as_float_ptr(d)
    rc = _lib.ct_stochastic(ph, pl, pc, pk, pd, close.size, int(k_period), int(d_period))
    if rc != 0:
        raise RuntimeError("ct_stochastic failed")
    return k, d

def cci(high, low, close, period):
    import numpy as np
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if high.shape != low.shape or high.shape != close.shape:
        raise ValueError("high, low, close must have same shape")
    out = np.zeros_like(close)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cci(ph, pl, pc, pout, close.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_cci failed")
    return out

def adx(high, low, close, period):
    import numpy as np
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if high.shape != low.shape or high.shape != close.shape:
        raise ValueError("high, low, close must have same shape")
    out = np.zeros_like(close)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_adx(ph, pl, pc, pout, close.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_adx failed")
    return out

def obv(price, volume):
    import numpy as np
    price = np.asarray(price, dtype=np.float32)
    volume = np.asarray(volume, dtype=np.float32)
    if price.shape != volume.shape:
        raise ValueError("price and volume must have same shape")
    out = np.zeros_like(price)
    _, pp = _as_float_ptr(price)
    _, pv = _as_float_ptr(volume)
    _, po = _as_float_ptr(out)
    rc = _lib.ct_obv(pp, pv, po, price.size)
    if rc != 0:
        raise RuntimeError("ct_obv failed")
    return out
