"""ctypes based Python bindings for the :mod:`tacuda` library.

The shared library must be built prior to importing this module.  The
bindings are generated from the public ``tacuda.h`` header in order to keep
the Python API in sync with the C++ implementation and to avoid large amounts
of manually written boilerplate.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def _load_lib() -> ctypes.CDLL:
    """Locate and load the ``tacuda`` shared library.

    The search order is:

    1. Explicit paths from the ``TACUDA_LIBRARY`` or
       ``TACUDA_LIBRARY_PATH``/``TACUDA_LIBRARY_DIR`` environment variables.
    2. System locations discovered via :func:`ctypes.util.find_library`.
    3. Common build directories relative to this file and the current working
       directory.
    """

    if sys.platform.startswith("win"):
        names = ["tacuda.dll"]
    elif sys.platform == "darwin":
        names = ["libtacuda.dylib", "tacuda.dylib"]
    else:
        names = ["libtacuda.so", "tacuda.so"]

    # Environment variables for explicit locations
    env_candidates: List[str] = []
    env = os.environ.get("TACUDA_LIBRARY")
    if env:
        env_candidates.append(env)

    env_dir = os.environ.get("TACUDA_LIBRARY_PATH") or os.environ.get(
        "TACUDA_LIBRARY_DIR"
    )
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
            path = os.path.join(base, n)
            if os.path.exists(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue

    raise OSError(
        "tacuda library not found. Build the project first or set "
        "TACUDA_LIBRARY[_PATH]."
    )


_lib = _load_lib()


def _as_float_array(arr: Any) -> Tuple[np.ndarray, ctypes.POINTER(ctypes.c_float)]:
    """Return ``arr`` as a contiguous ``float32`` array and pointer."""

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
    if arr.dtype != np.float32 or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


_C_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


def _parse_header() -> Dict[str, Dict[str, Any]]:
    """Parse ``tacuda.h`` and return a mapping of API specifications.

    Each entry in the mapping has the following structure::

        {
            "c_name": "ct_sma",             # name of the symbol in the C library
            "inputs": ["host_input"],        # names of pointer inputs
            "n_outputs": 1,                   # number of output arrays
            "params": [("period", c_int)],   # additional scalar parameters
        }
    """

    header = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "include", "tacuda.h")
    )
    spec: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(
        r"CTAPI_EXPORT\s+ctStatus_t\s+(ct_[a-zA-Z0-9_]+)\(([^)]*)\);",
        re.DOTALL,
    )
    with open(header, "r", encoding="utf-8") as fh:
        text = fh.read()

    for c_name, params in pattern.findall(text):
        params = [p.strip() for p in params.split(",") if p.strip()]
        inputs: List[str] = []
        outputs: List[str] = []
        scalars: List[Tuple[str, Any]] = []

        for p in params:
            tokens = p.split()
            name = tokens[-1]
            type_tokens = tokens[:-1]

            if "*" in p:
                if "const" in type_tokens:
                    inputs.append(name)
                else:
                    outputs.append(name)
            else:
                ctype: Any
                if "float" in type_tokens:
                    ctype = ctypes.c_float
                else:
                    ctype = ctypes.c_int
                scalars.append((name, ctype))

        # Remove the mandatory size parameter from scalars; it will be supplied
        # automatically by the wrapper based on the first input array.
        scalars = [(n, t) for n, t in scalars if n != "size"]

        py_name = c_name[3:]
        spec[py_name] = {
            "c_name": c_name,
            "inputs": inputs,
            "n_outputs": len(outputs),
            "params": scalars,
        }

    return spec


_SPEC = _parse_header()


def _build_wrapper(name: str, cfg: Dict[str, Any]) -> Callable[..., Any]:
    """Create and return a Python wrapper for ``name`` based on ``cfg``."""

    cfunc = getattr(_lib, cfg["c_name"])

    argtypes: List[Any] = []
    argtypes.extend([_C_FLOAT_PTR] * (len(cfg["inputs"]) + cfg["n_outputs"]))
    argtypes.append(ctypes.c_int)  # size
    for _, ctype in cfg["params"]:
        argtypes.append(ctype)
    cfunc.argtypes = argtypes
    cfunc.restype = ctypes.c_int

    in_args = cfg["inputs"]
    param_args = [p for p, _ in cfg["params"]]
    arg_list = in_args + param_args
    args_str = ", ".join(arg_list)

    def _impl(*args: Any) -> Any:
        arrays = args[: len(in_args)]
        params = args[len(in_args) :]
        if len(params) != len(param_args):
            raise TypeError("Incorrect number of parameters supplied")

        np_arrays: List[np.ndarray] = []
        c_pointers: List[_C_FLOAT_PTR] = []
        for arr in arrays:
            arr_np, ptr = _as_float_array(arr)
            np_arrays.append(arr_np)
            c_pointers.append(ptr)

        size = np_arrays[0].size
        for arr in np_arrays[1:]:
            if arr.size != size:
                raise ValueError("Input arrays must have the same size")

        outputs: List[Tuple[np.ndarray, _C_FLOAT_PTR]] = []
        for _ in range(cfg["n_outputs"]):
            out_arr = np.zeros(size, dtype=np.float32)
            outputs.append(_as_float_array(out_arr))

        c_args: List[Any] = c_pointers + [p for _, p in outputs]
        c_args.append(ctypes.c_int(size))
        for value, (_, ctype) in zip(params, cfg["params"]):
            c_args.append(ctype(value))

        rc = cfunc(*c_args)
        if rc != 0:
            raise RuntimeError(f"{cfg['c_name']} failed with code {rc}")

        result = [arr for arr, _ in outputs]
        if len(result) == 1:
            return result[0]
        return tuple(result)

    # Create a nicely named function using exec so that signature is clear.
    namespace: Dict[str, Any] = {"_impl": _impl}
    func_code = f"def {name}({args_str}):\n    return _impl({args_str})\n"
    exec(func_code, namespace)
    wrapper = namespace[name]
    wrapper.__doc__ = f"Auto-generated wrapper for {cfg['c_name']}"
    return wrapper


for _py_name, _cfg in _SPEC.items():
    globals()[_py_name] = _build_wrapper(_py_name, _cfg)


__all__ = sorted(_SPEC.keys())

