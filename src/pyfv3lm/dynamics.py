
from functools import partial
import numpy as np

from fv3jedilm_python_api import fv_grid_utils_nlm_mod

from pyfv3lm import constants
from pyfv3lm.array import xnp
from pyfv3lm.field_manager import MODEL_ATMOS
from pyfv3lm.grid import shallow_copy_grid, AxisType
from pyfv3lm.grid_utils import cubed_to_latlon
from pyfv3lm.nesting import setup_nested_grid_BCs
from pyfv3lm.regional import setup_regional_grid_BCs
from pyfv3lm.settings import default_float, MOIST_CAPPA, MULTI_GASES, USE_COND, SW_DYNAMICS
from pyfv3lm.tracer_manager import get_tracer_index
import py_mpp_numpy
from pyfv3lm.mpp import py_mpp
from pyfv3lm.transforms import jit
from jax import vjp

def fv_dynamics(local_grid_space, state, parameters, ak, bk, 
                bd, gridstruct, flagstruct, neststruct, idiag, 
                parent_grid, domain):
    
    rank = py_mpp_numpy.py_mpp_pe()
    print("Hello from fv_dynamics")
    print(f"rank {rank}: npx={parameters.npx}, npy={parameters.npy}, npz={parameters.npz}, is={bd.is_}, ie={bd.ie}, isd={bd.isd}, ied={bd.ied}, isc={bd.isc}, iec={bd.iec}")

    ua_copy = np.copy(state.ua)
    va_copy = np.copy(state.va)
    mode = 1
    fv_grid_utils_nlm_mod.cubed_to_latlon(state.u, state.v, state.ua, state.va, gridstruct, parameters.npx, parameters.npy, parameters.npz, mode,
                                          gridstruct.grid_type, domain, gridstruct.nested, flagstruct.c2l_ord, bd)

    domain_handle = np.array(domain._handle, dtype=np.int32)
    assert domain_handle[0] == domain._handle[0] and domain_handle[1] == domain._handle[1]
    domain_handle = py_mpp.hashable.pack_hashable_array(domain_handle)

    grid = shallow_copy_grid(gridstruct)
    ua_copy[...], va_copy[...] = cubed_to_latlon(state.u, state.v, grid, mode,
                    grid.grid_type, domain_handle, grid.nested, flagstruct.c2l_ord, local_grid_space)

    print(f"rank({rank}): ua arrays are close = {np.allclose(ua_copy, state.ua)}")
    print(f"rank({rank}): va arrays are close = {np.allclose(va_copy, state.va)}")
    
    return
   
    # Create temporaries
    dt2 = 0.5 * parameters.bdt
    nq = parameters.nq_tot - flagstruct.dnrts

    c_nx, c_ny = local_grid_space.compute_shape(AxisType.CENTER, AxisType.CENTER)
    d_nx, d_ny = local_grid_space.data_shape(AxisType.CENTER, AxisType.CENTER)
    
    te_2d = xnp.zeros((c_nx, c_ny), dtype=default_float)
    dp1 = xnp.zeros((d_nx, d_ny, parameters.npz), dtype=default_float)
    cappa = xnp.zeros((d_nx, d_ny, parameters.npz if MOIST_CAPPA else 1), dtype=default_float)

    if MULTI_GASES:
        kapad = parameters.kappa * xnp.ones((d_nx, d_ny, parameters.npz), dtype=default_float)


    # We call this BEFORE converting pt to virtual potential temperature, 
    # since we interpolate on (regular) temperature rather than theta.
    if gridstruct.nested or xnp.any(neststruct.child_grids):
        if USE_COND and MOIST_CAPPA:
            setup_nested_grid_BCs(parameters, state, ak, bk,
                                gridstruct, flagstruct, neststruct,
                                domain, parent_grid, bd, state.q_con, cappa)
        elif USE_COND:
            setup_nested_grid_BCs(parameters, state, ak, bk,
                                gridstruct, flagstruct, neststruct,
                                domain, parent_grid, bd, state.q_con)
        else:
            setup_nested_grid_BCs(parameters, state, ak, bk,
                                gridstruct, flagstruct, neststruct,
                                domain, parent_grid, bd)
        

    # For the regional domain set values valid the beginning of the
    # current large timestep at the boundary points of the pertinent
    # prognostic arrays.
    if flagstruct.regional:
        raise NotImplementedError("Regional grid BC setup branch not currently implemented.")
        # TODO: reinstate commented-out code below once `current_time_in_seconds` is handled.
        # reg_bc_update_time = current_time_in_seconds
        # if USE_COND and MOIST_CAPPA:
        #     setup_regional_grid_BCs(parameters, state, bd, reg_bc_update_time, state.q_con, cappa)
        # elif USE_COND:
        #     setup_regional_grid_BCs(parameters, state, bd, reg_bc_update_time, state.q_con)
        # elif MOIST_CAPPA:
        #     setup_regional_grid_BCs(parameters, state, bd, reg_bc_update_time, None, cappa)
        # else:
        #     setup_regional_grid_BCs(parameters, state, bd, reg_bc_update_time)

    if flagstruct.no_dycore:
        dynamics = no_dynamics
    else:
        if flagstruct.nwat == 0:
            sphum = 1
            cld_amt = -1   # to cause trouble if (mis)used
        else:
            sphum   = get_tracer_index(MODEL_ATMOS, 'sphum')
            liq_wat = get_tracer_index(MODEL_ATMOS, 'liq_wat')
            ice_wat = get_tracer_index(MODEL_ATMOS, 'ice_wat')
            rainwat = get_tracer_index(MODEL_ATMOS, 'rainwat')
            snowwat = get_tracer_index(MODEL_ATMOS, 'snowwat')
            graupel = get_tracer_index(MODEL_ATMOS, 'graupel')
            cld_amt = get_tracer_index(MODEL_ATMOS, 'cld_amt')

        theta_d = get_tracer_index (MODEL_ATMOS, 'theta_d')

        if SW_DYNAMICS:
            dynamics = sw_dynamics
        else:
            dynamics = gcm_dynamics

    dynamics()

    state.ua, state.va = cubed_to_latlon(state.u, state.v, grid, mode, grid.grid_type, domain_handle,
                                         grid.nested, flagstruct.c2l_ord, local_grid_space)

def no_dynamics():
    pass

def sw_dynamics():
    raise NotImplementedError("Function `sw_dynamics` not currently implemented")

def gcm_dynamics():
    raise NotImplementedError("Function `gcm_dynamics` not currently implemented")