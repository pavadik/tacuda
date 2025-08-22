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
_lib.ct_sar.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                        ctypes.c_float, ctypes.c_float]
_lib.ct_sar.restype  = ctypes.c_int
_lib.ct_aroon.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ct_aroon.restype  = ctypes.c_int
_lib.ct_ultosc.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ct_ultosc.restype  = ctypes.c_int
_lib.ct_trange.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_trange.restype  = ctypes.c_int
_lib.ct_sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_sum.restype  = ctypes.c_int
_lib.ct_t3.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]
_lib.ct_t3.restype  = ctypes.c_int
_lib.ct_trima.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_lib.ct_trima.restype  = ctypes.c_int
_lib.ct_stochrsi.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ct_stochrsi.restype  = ctypes.c_int
_lib.ct_cdl_doji.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_doji.restype  = ctypes.c_int
_lib.ct_cdl_hammer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_hammer.restype  = ctypes.c_int
_lib.ct_cdl_inverted_hammer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_inverted_hammer.restype  = ctypes.c_int
_lib.ct_cdl_bullish_engulfing.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_bullish_engulfing.restype  = ctypes.c_int
_lib.ct_cdl_bearish_engulfing.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_bearish_engulfing.restype  = ctypes.c_int
_lib.ct_cdl_three_white_soldiers.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_three_white_soldiers.restype  = ctypes.c_int
_lib.ct_cdl_abandoned_baby.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_abandoned_baby.restype  = ctypes.c_int
_lib.ct_cdl_advance_block.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_advance_block.restype  = ctypes.c_int
_lib.ct_cdl_belt_hold.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_belt_hold.restype  = ctypes.c_int
_lib.ct_cdl_breakaway.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_breakaway.restype  = ctypes.c_int
_lib.ct_cdl_two_crows.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                  ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_two_crows.restype  = ctypes.c_int
_lib.ct_cdl_three_black_crows.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_three_black_crows.restype  = ctypes.c_int
_lib.ct_cdl_three_inside.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_three_inside.restype  = ctypes.c_int
_lib.ct_cdl_three_line_strike.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_three_line_strike.restype  = ctypes.c_int
_lib.ct_cdl_three_stars_in_south.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_three_stars_in_south.restype  = ctypes.c_int
_lib.ct_cdl_closing_marubozu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_closing_marubozu.restype  = ctypes.c_int
_lib.ct_cdl_conceal_baby_swallow.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_conceal_baby_swallow.restype  = ctypes.c_int
_lib.ct_cdl_counterattack.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_counterattack.restype  = ctypes.c_int
_lib.ct_cdl_dark_cloud_cover.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.ct_cdl_dark_cloud_cover.restype  = ctypes.c_int

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

def sar(high, low, step=0.02, max_acc=0.2):
    import numpy as np
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    if high.shape != low.shape:
        raise ValueError("high and low must have same shape")
    out = np.zeros_like(high)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, po = _as_float_ptr(out)
    rc = _lib.ct_sar(ph, pl, po, high.size, float(step), float(max_acc))
    if rc != 0:
        raise RuntimeError("ct_sar failed")
    return out

def aroon(high, low, up_period, down_period):
    import numpy as np
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    up = np.zeros_like(high)
    down = np.zeros_like(high)
    osc = np.zeros_like(high)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pu = _as_float_ptr(up)
    _, pd = _as_float_ptr(down)
    _, po = _as_float_ptr(osc)
    rc = _lib.ct_aroon(ph, pl, pu, pd, po, high.size, int(up_period), int(down_period))
    if rc != 0:
        raise RuntimeError("ct_aroon failed")
    return up, down, osc

def ultosc(high, low, close, short_period, medium_period, long_period):
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
    _, po = _as_float_ptr(out)
    rc = _lib.ct_ultosc(ph, pl, pc, po, close.size,
                        int(short_period), int(medium_period), int(long_period))
    if rc != 0:
        raise RuntimeError("ct_ultosc failed")
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

def cdl_doji(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_doji(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_doji failed")
    return out

def cdl_hammer(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_hammer(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_hammer failed")
    return out

def cdl_inverted_hammer(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_inverted_hammer(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_inverted_hammer failed")
    return out

def cdl_bullish_engulfing(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_bullish_engulfing(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_bullish_engulfing failed")
    return out

def cdl_bearish_engulfing(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_bearish_engulfing(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_bearish_engulfing failed")
    return out

def cdl_three_white_soldiers(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_three_white_soldiers(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_three_white_soldiers failed")
    return out

def cdl_abandoned_baby(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_abandoned_baby(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_abandoned_baby failed")
    return out

def cdl_advance_block(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_advance_block(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_advance_block failed")
    return out

def cdl_belt_hold(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_belt_hold(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_belt_hold failed")
    return out

def cdl_breakaway(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_breakaway(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_breakaway failed")
    return out

def cdl_two_crows(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_two_crows(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_two_crows failed")
    return out

def cdl_three_black_crows(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_three_black_crows(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_three_black_crows failed")
    return out

def cdl_three_inside(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_three_inside(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_three_inside failed")
    return out

def cdl_three_line_strike(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_three_line_strike(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_three_line_strike failed")
    return out

def cdl_three_stars_in_south(open, high, low, close):
    import numpy as np
    open = np.asarray(open, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    if open.shape != high.shape or open.shape != low.shape or open.shape != close.shape:
        raise ValueError("open, high, low, close must have same shape")
    out = np.zeros_like(open)
    _, po = _as_float_ptr(open)
    _, ph = _as_float_ptr(high)
    _, pl = _as_float_ptr(low)
    _, pc = _as_float_ptr(close)
    _, pout = _as_float_ptr(out)
    rc = _lib.ct_cdl_three_stars_in_south(po, ph, pl, pc, pout, open.size)
    if rc != 0:
        raise RuntimeError("ct_cdl_three_stars_in_south failed")
    return out

def trange(high, low, close):
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
    _, po = _as_float_ptr(out)
    rc = _lib.ct_trange(ph, pl, pc, po, close.size)
    if rc != 0:
        raise RuntimeError("ct_trange failed")
    return out

def summation(x, period):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    _, px = _as_float_ptr(x)
    _, po = _as_float_ptr(out)
    rc = _lib.ct_sum(px, po, x.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_sum failed")
    return out

def t3(x, period, v_factor):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    _, px = _as_float_ptr(x)
    _, po = _as_float_ptr(out)
    rc = _lib.ct_t3(px, po, x.size, int(period), float(v_factor))
    if rc != 0:
        raise RuntimeError("ct_t3 failed")
    return out

def trima(x, period):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    _, px = _as_float_ptr(x)
    _, po = _as_float_ptr(out)
    rc = _lib.ct_trima(px, po, x.size, int(period))
    if rc != 0:
        raise RuntimeError("ct_trima failed")
    return out

def stochrsi(x, rsi_period, k_period, d_period):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    k = np.zeros_like(x)
    d = np.zeros_like(x)
    _, px = _as_float_ptr(x)
    _, pk = _as_float_ptr(k)
    _, pd = _as_float_ptr(d)
    rc = _lib.ct_stochrsi(px, pk, pd, x.size, int(rsi_period), int(k_period), int(d_period))
    if rc != 0:
        raise RuntimeError("ct_stochrsi failed")
    return k, d


__all__ = [
    "sma",
    "wma",
    "momentum",
    "macd_line",
    "rsi",
    "atr",
    "stochastic",
    "cci",
    "adx",
    "sar",
    "aroon",
    "ultosc",
    "obv",
    "cdl_doji",
    "cdl_hammer",
    "cdl_inverted_hammer",
    "cdl_bullish_engulfing",
    "cdl_bearish_engulfing",
    "cdl_three_white_soldiers",
    "cdl_abandoned_baby",
    "cdl_advance_block",
    "cdl_belt_hold",
    "cdl_breakaway",
    "cdl_two_crows",
    "cdl_three_black_crows",
    "cdl_three_inside",
    "cdl_three_line_strike",
    "cdl_three_stars_in_south",
    "trange",
    "summation",
    "t3",
    "trima",
    "stochrsi",
]
