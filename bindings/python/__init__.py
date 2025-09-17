"""ctypes based Python bindings for the :mod:`tacuda` library.

The shared library must be built prior to importing this module.  The
bindings are generated from the public ``tacuda.h`` header in order to keep
the Python API in sync with the C++ implementation and to avoid large amounts
of manually written boilerplate.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import inspect
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from . import _generated


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
_SPEC = _generated.FUNCTIONS


def _scalar_ctype(name: str) -> Any:
    mapping: Dict[str, Any] = {
        "int": ctypes.c_int,
        "float": ctypes.c_float,
        "ctMaType_t": ctypes.c_int,
        "cudaStream_t": ctypes.c_void_p,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise KeyError(f"Unsupported scalar type '{name}' in tacuda bindings") from exc


def _prepare_stream(value: Any) -> ctypes.c_void_p:
    if value is None:
        return ctypes.c_void_p(0)
    if isinstance(value, ctypes.c_void_p):
        return value
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    raise TypeError("Stream parameters must be integers or ctypes.c_void_p instances")


def _bind_function(name: str, spec: _generated.FunctionSpec) -> None:
    cfunc = getattr(_lib, spec.c_name)

    argtypes: List[Any] = []
    argtypes.extend([_C_FLOAT_PTR] * (len(spec.inputs) + len(spec.outputs)))
    argtypes.append(ctypes.c_int)
    for scalar in spec.scalars:
        argtypes.append(_scalar_ctype(scalar.ctype))
    cfunc.argtypes = argtypes
    cfunc.restype = ctypes.c_int

    parameters: List[inspect.Parameter] = []
    for input_name in spec.inputs:
        parameters.append(
            inspect.Parameter(input_name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )

    for scalar in spec.scalars:
        default = inspect._empty
        if scalar.ctype == "cudaStream_t" and scalar.default is not None:
            default = None
        parameters.append(
            inspect.Parameter(
                scalar.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
            )
        )

    signature = inspect.Signature(parameters)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        np_inputs: List[np.ndarray] = []
        c_inputs: List[_C_FLOAT_PTR] = []
        for name_ in spec.inputs:
            arr_np, ptr = _as_float_array(bound.arguments[name_])
            np_inputs.append(arr_np)
            c_inputs.append(ptr)
        if not np_inputs:
            raise ValueError(f"{name} requires at least one input array")

        size = np_inputs[0].size
        for arr in np_inputs[1:]:
            if arr.size != size:
                raise ValueError("Input arrays must have the same size")

        outputs: List[Tuple[np.ndarray, _C_FLOAT_PTR]] = []
        for _ in spec.outputs:
            out_arr = np.zeros(size, dtype=np.float32)
            outputs.append(_as_float_array(out_arr))

        c_args: List[Any] = []
        c_args.extend(c_inputs)
        c_args.extend(ptr for _, ptr in outputs)
        c_args.append(ctypes.c_int(size))

        for scalar in spec.scalars:
            value = bound.arguments[scalar.name]
            if scalar.ctype == "cudaStream_t":
                c_args.append(_prepare_stream(value))
            else:
                ctype = _scalar_ctype(scalar.ctype)
                c_args.append(ctype(value))

        rc = cfunc(*c_args)
        if rc != 0:
            raise RuntimeError(f"{spec.c_name} failed with code {rc}")

        result = [arr for arr, _ in outputs]
        if len(result) == 1:
            return result[0]
        if not result:
            return None
        return tuple(result)

    wrapper.__name__ = name
    wrapper.__doc__ = f"Python wrapper for {spec.c_name}"
    wrapper.__signature__ = signature
    globals()[name] = wrapper


for _py_name, _spec in _SPEC.items():
    _bind_function(_py_name, _spec)


__all__ = tuple(sorted(_SPEC))

