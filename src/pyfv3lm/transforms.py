from pyfv3lm.settings import backend

if backend == "jax":
    from jax import jit, vmap
    from jax.lax import fori_loop
elif backend == "numpy":
    from pyfv3lm.numpy_transforms import jit_numpy as jit
    from pyfv3lm.numpy_transforms import vmap_numpy as vmap
    from pyfv3lm.numpy_transforms import fori_loop_numpy as fori_loop
else:
    raise ValueError(f"Unsupported backend, {backend}.")