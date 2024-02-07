import numpy as np


def _array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

def _canonicalize_in_axes(in_axes: int | tuple | None, *args):
    nargs = len(args)

    if in_axes == None:
        in_axes = 0
    
    if isinstance(in_axes, int):
        in_axes = tuple([in_axes] * nargs)

    assert isinstance(in_axes, tuple)

    assert len(in_axes) == nargs
    for axis, arg in zip(in_axes, args):
        assert 0 <= axis and axis <= arg.ndim

    return in_axes

def _canonicalize_out_axes(out_axes: int | tuple | None, *results):
    nresults = len(results)

    if out_axes == None:
        out_axes = 0
    
    if isinstance(out_axes, int):
        out_axes = tuple([out_axes] * nresults)

    assert isinstance(out_axes, tuple)
    assert len(out_axes) == nresults
    return out_axes
    
def _get_mapped_axis_size(in_axes: tuple, *args):
    assert isinstance(in_axes, tuple) and len(in_axes) == len(args)

    in_axes_size = []
    for in_axis, arg in zip(in_axes, args):
        in_axes_size.append(np.shape(arg)[in_axis])
    for size in in_axes_size:
        assert size == in_axes_size[0]
    
    return in_axes_size[0]

def vmap_numpy(fun, in_axes, out_axes):
    """
    Creates a function which maps `fun` over selected axes. It is intended as a Numpy alternative to `jax.numpy`, however, it has a reduced interface and can only work on functions with a flat list of arrays as parameters, not over arbitrary PyTrees as in the `jax.vmap` case. 
    """
    
    def mapped_func(*args):

        for arg in args:
            print(np.shape(arg))

        in_axes_ = _canonicalize_in_axes(in_axes, *args)
        mapped_axis_size = _get_mapped_axis_size(in_axes_, *args)

        results = []
        for i in range(mapped_axis_size):
            args_slices = [np.squeeze(_array_slice(arg, in_axis, i, i+1)) for in_axis, arg in zip(in_axes_, args)]
            result_slices = fun(*args_slices)
            if results == []:
                results = [[e] for e in result_slices]
            else:
                for i, e in enumerate(result_slices):
                    results[i].append(e)

        out_axes_ = _canonicalize_out_axes(out_axes, *results)
        return tuple(np.stack(res, axis=out_axis) for out_axis, res in zip(out_axes_, results))
    
    return mapped_func


def fori_loop_numpy(lower, upper, body_fun, init_val, *, unroll=None):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def jit_numpy(fun, in_shardings=None, out_shardings=None, static_argnums=None, static_argnames=None, donate_argnums=None, donate_argnames=None, keep_unused=False, device=None, backend=None, inline=False, abstracted_axes=None):
        return fun