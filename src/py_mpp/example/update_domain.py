from functools import partial

import py_mpp_jax
import py_mpp_numpy


import numpy as np
import jax 
jax.config.update("jax_enable_x64", True)
from jax import jit
from jax import numpy as jnp



def print_on_root(pe, root, msg):
    if pe == root:
        print(msg)


domain_handle = py_mpp_numpy.py_mpp_allocate_domain2d()
print(f"domain allocated")

root = 0

py_mpp_numpy.py_mpp_init()
py_mpp_numpy.py_mpp_domains_init()
py_mpp_numpy.py_mpp_domains_set_stack_size(4000000)
py_mpp_numpy.py_mpp_set_root_pe(root)

pe = py_mpp_numpy.py_mpp_pe()
npes = py_mpp_numpy.py_mpp_pe()

gnx = 12
gny = 12
global_indices = np.array([1,gnx,1,gny], dtype=np.int32)
npes = 4
gridtype = py_mpp_numpy.parameters.AGRID

layout = py_mpp_numpy.py_mpp_define_layout2d(global_indices,npes)
print_on_root(pe, root, f"layout defined")
print_on_root(pe, root, f"layout={layout}")


xhalo = 1
yhalo = 1
print_info = 1
py_mpp_numpy.py_mpp_define_domains2d(global_indices, layout, domain_handle, xhalo, yhalo, print_info)
print_on_root(pe, root, f"domain defined")

compute_indices = py_mpp_numpy.py_mpp_get_compute_domain2d(domain_handle)
print_on_root(pe, root, f"compute_indices={compute_indices}")

data_indices = py_mpp_numpy.py_mpp_get_data_domain2d(domain_handle)
print_on_root(pe, root, f"data_indices={data_indices}")

local_dnx = (data_indices[1] - data_indices[0]) + 1
local_dny = (data_indices[3] - data_indices[2]) + 1
u = np.zeros((local_dnx,local_dny,2), dtype=np.float64, order='F')
u[1:-1,1:-1,0] = 1
u[1:-1,1:-1,1] = 2
v = np.zeros((local_dnx,local_dny,2), dtype=np.float64, order='F')
v[1:-1,1:-1,0] = 3
v[1:-1,1:-1,1] = 4
py_mpp_numpy.py_mpp_sync()
print_on_root(pe, root, f"u[:,:,0]=\n{u[:,:,0]}")

if True:
    u, v = py_mpp_numpy.py_mpp_update_domain2d_r8_3dv(u, v, domain_handle, gridtype)
else:
    @jit
    def call_py_mpp_update_domain2d_r8_3dv(u, v, domain_handle, gridtype):
        return py_mpp_jax.py_mpp_update_domain2d_r8_3dv(u, v, domain_handle, gridtype)
    u, v = call_py_mpp_update_domain2d_r8_3dv(u, v, domain_handle, gridtype)

print_on_root(pe, root, f"u[:,:,0]=\n{u[:,:,0]}")

py_mpp_numpy.py_mpp_sync()

py_mpp_numpy.py_mpp_domains_exit()
py_mpp_numpy.py_mpp_exit()

py_mpp_numpy.py_mpp_deallocate_domain2d(domain_handle)
print_on_root(pe, root, f"domain deallocated")