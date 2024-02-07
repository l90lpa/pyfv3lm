from pyfv3lm.settings import backend

if backend == "jax":
    import py_mpp_jax as py_mpp
elif backend == "numpy":
    import py_mpp_numpy as py_mpp
else:
    raise ValueError(f"Unsupported backend, {backend}.")