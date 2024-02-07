from settings import backend

def shift_slice(slice_: slice, offset: int):
    assert isinstance(offset, int)
    return slice(slice_.start+offset if slice_.start else None,
                 slice_.stop+offset if slice_.stop else None,
                 slice_.step)


def update_jax_array(array, *, at, to):
    return array.at[at].set(to)


def update_numpy_array(array, *, at, to):
    array[at] = to
    return array


if backend == "jax":
    from jax import config
    config.update("jax_enable_x64", True)
    import jax.numpy as xnp
    update = update_jax_array
    from jax import typing
    xnp.typing = typing
elif backend == "numpy":
    import numpy as xnp
    update = update_numpy_array
else:
    raise ValueError(f"Unsupported backend, {backend}.")