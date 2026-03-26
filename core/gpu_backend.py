"""Optional CuPy GPU backend. Falls back to NumPy if CuPy unavailable."""
try:
    import cupy as cp
    import cupy.fft as fft_module
    HAS_GPU = True
except ImportError:
    import numpy as cp  # type: ignore  # NumPy as drop-in
    import numpy.fft as fft_module  # type: ignore
    HAS_GPU = False


def to_gpu(arr):
    """Transfer array to GPU (no-op if CuPy unavailable)."""
    return cp.asarray(arr) if HAS_GPU else arr


def to_cpu(arr):
    """Transfer array from GPU to NumPy (no-op if CuPy unavailable)."""
    return cp.asnumpy(arr) if HAS_GPU else arr


def fft2(arr):
    return fft_module.fft2(arr)


def ifft2(arr):
    return fft_module.ifft2(arr)


def fftshift(arr):
    return fft_module.fftshift(arr)
