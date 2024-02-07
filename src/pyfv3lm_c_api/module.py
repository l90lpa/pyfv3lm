from pyfv3lm_c_api import ffi

import numpy as np

from fv3jedilm_python_api import fv_arrays_nlm_mod as fv_types
from fv3jedilm_python_api import mpp_domains_mod

from pyfv3lm.dynamics import fv_dynamics
from pyfv3lm.grid import to_local_grid_space, AxisType
from pyfv3lm.state import State, Parameters

@ffi.def_extern()
def pyfv3lm_fv_dynamics_r8(npx, npy, npz, nq_tot, ng, bdt, consv_te, 
                           fill, reproduce_sum, kappa, cp_air, zvir, 
                           ptop, ks, ncnst, n_split, q_split, u, v,
                           w, delz, hydrostatic, pt, delp, q,
                           ps, pe, pk, peln, pkz, phis,
                           q_con, omga, ua, va, uc, vc,
                           ak, bk, mfx, mfy, cx, cy,
                           ze0, hybrid_z, gridstruct_handle, flagstruct_handle, neststruct_handle, 
                           idiag_handle, c_bd, parent_grid_handle, domain_handle,
                           time_total):
    
    print("Hello from pyfv3lm_fv_dynamics")

    # Create Fortran object wrappers
    bd = create_fv_grid_bounds_type(c_bd)
    gridstruct = fv_types.fv_grid_type.from_handle([gridstruct_handle[0], gridstruct_handle[1]])
    flagstruct = fv_types.fv_flags_type.from_handle([flagstruct_handle[0], flagstruct_handle[1]])
    neststruct = fv_types.fv_nest_type.from_handle([neststruct_handle[0], neststruct_handle[1]])
    idiag = fv_types.fv_diag_type.from_handle([idiag_handle[0], idiag_handle[1]])
    parent_grid = fv_types.fv_atmos_type.from_handle([parent_grid_handle[0], parent_grid_handle[1]])
    domain = mpp_domains_mod.domain2D.from_handle([domain_handle[0], domain_handle[1]])

    # Create local grid space
    local_grid_space = to_local_grid_space(c_bd, npx, npy, npz)
    
    # Pack state
    state = state_from_fortran(local_grid_space, npz, ncnst, u, v, w, pt, delp, q, delz, ze0, ps, pe, pk,
                               peln, pkz, q_con, omga, phis, uc, vc, ua, va, mfx, mfy, cx, cy)

    # Pack parameters
    parameters = Parameters(bdt, consv_te, kappa, cp_air, zvir, ptop, time_total, npx, npy, npz, nq_tot,
                            ng, ks, ncnst, n_split, q_split, fill, reproduce_sum, hydrostatic, hybrid_z)
    z_outer = npz + 1
    ak_ref = as_ndarray(ak, (z_outer,), order='F')
    bk_ref = as_ndarray(bk, (z_outer,), order='F')

    fv_dynamics(local_grid_space, state, parameters, ak_ref, bk_ref, 
                bd, gridstruct, flagstruct, neststruct, idiag, 
                parent_grid, domain)


### Helper functions ###   


def state_from_fortran(local_grid_space, npz, ncnst, u, v, w, pt, delp, q, delz, ze0, ps, pe, pk,
                       peln, pkz, q_con, omga, phis, uc, vc, ua, va, mfx, mfy, cx, cy):
    
    (x_center, y_center) = local_grid_space.data_shape(AxisType.CENTER, AxisType.CENTER)
    (x_outer, y_outer) = local_grid_space.data_shape(AxisType.OUTER, AxisType.OUTER)
    z_center = npz
    z_outer = npz + 1

    u_ref     = as_ndarray(u,    (x_center, y_outer, z_center), order='F')
    v_ref     = as_ndarray(v,    (x_outer, y_center, z_center), order='F')
    w_ref     = as_ndarray(w,    (x_outer, y_center, z_center), order='F')
    pt_ref    = as_ndarray(pt,   (x_center, y_center, z_center), order='F')
    delp_ref  = as_ndarray(delp, (x_center, y_center, z_center), order='F')
    q_ref     = as_ndarray(q,    (x_center, y_center, z_center, ncnst), order='F')
    delz_ref  = as_ndarray(delz, (x_center, y_center, z_center), order='F')
    # TODO: check ze0 shape
    ze0_ref   = as_ndarray(ze0,  (x_center, y_outer, z_center), order='F')
    ps_ref    = as_ndarray(ps,   (x_center, y_center), order='F')
    pe_ref    = as_ndarray(pe,   (x_center+1, z_outer, y_center+1), order='F')
    pk_ref    = as_ndarray(pk,   (x_center, y_center, z_outer), order='F')
    # The order of shape arguments is intentionally, (x, z, y), for 'pe' and 
    # 'peln'. And 'pe' has and additional +1 for the first and third position.
    peln_ref  = as_ndarray(peln, (x_center, z_outer, y_center), order='F')
    pkz_ref   = as_ndarray(pkz,  (x_center, y_center, z_center), order='F')
    q_con_ref = as_ndarray(q_con, (x_center, y_center, z_center), order='F')
    omga_ref  = as_ndarray(omga, (x_center, y_center, z_center), order='F')
    phis_ref  = as_ndarray(phis, (x_center, y_center), order='F')
    uc_ref    = as_ndarray(uc,   (x_outer, y_center, z_center), order='F')
    vc_ref    = as_ndarray(vc,   (x_center, y_outer, z_center), order='F')
    ua_ref    = as_ndarray(ua,   (x_center, y_center, z_center), order='F')
    va_ref    = as_ndarray(va,   (x_center, y_center, z_center), order='F')
    mfx_ref   = as_ndarray(mfx,  (x_outer, y_center, z_center), order='F')
    mfy_ref   = as_ndarray(mfy,  (x_center, y_outer, z_center), order='F')
    cx_ref    = as_ndarray(cx,   (x_outer, y_center, z_center), order='F')
    cy_ref    = as_ndarray(cy,   (x_center, y_outer, z_center), order='F')

    return State(u_ref, v_ref, w_ref, pt_ref, delp_ref, q_ref, delz_ref, ze0_ref, ps_ref, pe_ref, pk_ref, peln_ref, pkz_ref,
                 q_con_ref, omga_ref, phis_ref, uc_ref, vc_ref, ua_ref, va_ref, mfx_ref, mfy_ref, cx_ref, cy_ref)
    

def as_ndarray(ptr, shape: tuple, order:{'C','F','A'}) -> np.ndarray:
    """
    Create a non-owning NDArray from a C ptr to a memory buffer.
    """
    length = np.prod(shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    array = np.frombuffer(
        ffi.buffer(ptr, length * ffi.sizeof(c_type)),
        dtype=np.dtype(c_type),
        count=-1,
        offset=0,
    ).reshape(shape, order=order)
    return array


def copy_to_buffer(ptr, array: np.ndarray):
    """
    Memcopy the contents of an NDArray into buffer accessed through ptr.
    """
    length = np.prod(array.shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    ffi.memmove(ptr, np.ravel(array), length * ffi.sizeof(c_type))


def create_fv_grid_bounds_type(c_bd):
    bd = fv_types.fv_grid_bounds_type()
    bd.is_ = c_bd.Is
    bd.ie  = c_bd.Ie
    bd.js  = c_bd.Js
    bd.je  = c_bd.Je
    bd.isd = c_bd.Isd
    bd.ied = c_bd.Ied
    bd.jsd = c_bd.Jsd
    bd.jed = c_bd.Jed
    bd.isc = c_bd.Isc
    bd.iec = c_bd.Iec
    bd.jsc = c_bd.Jsc
    bd.jec = c_bd.Jec
    bd.ng  = c_bd.ng
    return bd