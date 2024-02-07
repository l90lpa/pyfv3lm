import pytest

from jax import vjp
import jax.numpy as jnp
import numpy as np
import json

from pyfv3lm.grid import Grid, LocalGridSpace
from pyfv3lm.grid_utils import cubed_to_latlon
from pyfv3lm.json_utils import ndarray_decoder
from pyfv3lm.mpp import py_mpp


def test_cubed_to_latlon_a():

    rank = 3
    
    with open(f"./data/cubed_to_latlon_input_mpi_rank{rank}.json", "r") as f:
        thawed = json.loads(f.read())

        u = ndarray_decoder(thawed['u'])
        v = ndarray_decoder(thawed['v'])
        ua = ndarray_decoder(thawed['ua'])
        va = ndarray_decoder(thawed['va'])
        gridstruct = Grid.from_dict(thawed['gridstruct'])
        npx = thawed['npx']
        npy = thawed['npy']
        npz = thawed['npz']
        mode = thawed['mode']
        domain = thawed['domain']
        grid_type = thawed['grid_type']
        nested = thawed['nested']
        c2l_ord = thawed['c2l_ord']
        local_grid_space = LocalGridSpace.from_dict(thawed['local_grid_space'])

    # Currently we don't have a way fo caching domain properly and so here we are forcing the
    # mode 0 so that a domain update ins't invoked - and setting domain to null.
    mode = 0
    domain = py_mpp.hashable.pack_hashable_array(jnp.array([0,0], dtype=jnp.int32))

    with open(f"./data/cubed_to_latlon_output_mpi_rank{rank}.json", "r") as f:
        thawed = json.loads(f.read())

        ref_ua = ndarray_decoder(thawed['ua'])
        ref_va = ndarray_decoder(thawed['va'])

    ua, va = cubed_to_latlon(u, v, gridstruct, mode,
                             grid_type, domain, nested, c2l_ord, local_grid_space)

    print("diff = ", np.max(np.abs(ua[:,:,0] - ref_ua[:,:,0])))
    print("diff = ", np.max(np.abs(va[:,:,0] - ref_va[:,:,0])))

    assert np.max(np.abs(ua[:,:,0] - ref_ua[:,:,0])) < 1e-12
    assert np.max(np.abs(va[:,:,0] - ref_va[:,:,0])) < 1e-12
    