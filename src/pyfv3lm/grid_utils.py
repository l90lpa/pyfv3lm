from functools import partial

from pyfv3lm.array import update, xnp, shift_slice as shift
from pyfv3lm.transforms import jit, vmap
from pyfv3lm.mpp import py_mpp
from pyfv3lm.grid import AxisType
from pyfv3lm.settings import default_float


@partial(jit, static_argnames=['domain', 'grid_type', 'nested', 'c2l_ord', 'mode', 'local_grid_space'])
def cubed_to_latlon(u, v, gridstruct, mode, 
                    grid_type, domain, nested, c2l_ord, local_grid_space):
    if c2l_ord == 2:
        return c2l_ord2(u, v, gridstruct, grid_type, local_grid_space, False)
    else:
        return c2l_ord4(u, v, gridstruct, grid_type, domain, nested, mode, local_grid_space)


def c2l_ord2(u, v, gridstruct, grid_type, local_grid_space, do_halo):
    
    nx_center, ny_center = local_grid_space.compute_shape(AxisType.CENTER, AxisType.CENTER)
    Is = local_grid_space.Is()
    Ie = local_grid_space.Ie()
    Js = local_grid_space.Js()
    Je = local_grid_space.Je()

    # TODO: extract `n_halo` to local_grid_space or some other class
    n_halo = 3

    if do_halo:
        Is = Is-1
        Ie = Ie+1
        Js = Js-1
        Je = Je+1

    dx = gridstruct.dx
    dy = gridstruct.dy

    # TODO: remove code slice gridstruct.a11, etc once they are created with a 
    #       shape equal to the cell center compute domain. Problem: in the 
    #       Fortran FV3 code a11, a12, etc are created with a shape equal to the
    #       cell center compute domain plus a halo of 1, however they seem to 
    #       only ever be read from their cell center compute domain portion. Here
    #       we are slicing them to the cell center compute domain to canonicalize
    #       them.
    if xnp.shape(gridstruct.a11) == (nx_center + 2, ny_center + 2):
        a11 = gridstruct.a11[1:-1, 1:-1]
        a12 = gridstruct.a12[1:-1, 1:-1]
        a21 = gridstruct.a21[1:-1, 1:-1]
        a22 = gridstruct.a22[1:-1, 1:-1]
    elif xnp.shape(gridstruct.a11) == (nx_center, ny_center):
        a11 = gridstruct.a11
        a12 = gridstruct.a12
        a21 = gridstruct.a21
        a22 = gridstruct.a22
    else:
        # Unexpected case
        assert False

    if grid_type < 4:
        def cubed_sphere_transform(u, v):
            I = slice(Is,Ie)
            J = slice(Js,Je+1)
            wu = u[I,J] * dx[I,J]

            I = slice(Is,Ie+1)
            J = slice(Js,Je)
            wv = v[I,J] * dy[I,J]

            I = slice(Is,Ie)
            J = slice(Js,Je)
            # Co-variant to Co-variant "vorticity-conserving" interpolation
            u1 = 2.0 * (wu[:,:-1] + wu[:,1:]) / (dx[I,J]+dx[I,shift(J,1)])
            v1 = 2.0 * (wv[:-1,:] + wv[1:,:]) / (dy[I,J]+dy[shift(I,1),J])
            # Cubed (cell center co-variant winds) to lat-lon:
            ua = a11*u1 + a12*v1
            va = a21*u1 + a22*v1

            ua = xnp.pad(ua, n_halo)
            va = xnp.pad(va, n_halo)
            return ua, va

        transform = cubed_sphere_transform
    else:
        def cartesian_geometry_transform(u, v):
            I = slice(Is,Ie)
            J = slice(Js,Je)
            ua = 0.5*(u[I,J] + u[I,shift(J,1)])
            va = 0.5*(v[I,J] + v[shift(I,1),J])

            ua = xnp.pad(ua, n_halo)
            va = xnp.pad(va, n_halo)
            return ua, va

        transform = cartesian_geometry_transform
    
    vtransform = vmap(transform, 2, 2)
    ua, va = vtransform(u, v)

    return ua, va


def c2l_ord4(u, v, gridstruct, grid_type, domain, nested, mode, local_grid_space):
    print("c2l_ord4")
    
    nx_center, ny_center = local_grid_space.compute_shape(AxisType.CENTER, AxisType.CENTER)
    Is = local_grid_space.Is()
    Ie = local_grid_space.Ie()
    Js = local_grid_space.Js()
    Je = local_grid_space.Je()
    west_edge  = local_grid_space.west_edge 
    east_edge  = local_grid_space.east_edge 
    south_edge = local_grid_space.south_edge
    north_edge = local_grid_space.north_edge
    
    # TODO: extract `n_halo` to local_grid_space or some other class
    n_halo = 3

    # 4-pt Lagrange interpolation
    a1 = 0.5625
    a2 = -0.0625
    c1 = 1.125
    c2 = -0.125

    dx = gridstruct.dx
    dy = gridstruct.dy

    # TODO: remove code slice gridstruct.a11, etc once they are created with a 
    #       shape equal to the cell center compute domain. Problem: in the 
    #       Fortran FV3 code a11, a12, etc are created with a shape equal to the
    #       cell center compute domain plus a halo of 1, however they seem to 
    #       only ever be read from their cell center compute domain portion. Here
    #       we are slicing them to the cell center compute domain to canonicalize
    #       them.
    if xnp.shape(gridstruct.a11) == (nx_center + 2, ny_center + 2):
        a11 = gridstruct.a11[1:-1, 1:-1]
        a12 = gridstruct.a12[1:-1, 1:-1]
        a21 = gridstruct.a21[1:-1, 1:-1]
        a22 = gridstruct.a22[1:-1, 1:-1]
    elif xnp.shape(gridstruct.a11) == (nx_center, ny_center):
        a11 = gridstruct.a11
        a12 = gridstruct.a12
        a21 = gridstruct.a21
        a22 = gridstruct.a22
    else:
        # Unexpected case
        assert False
 
    if mode > 0:
        py_mpp.py_mpp_update_domain2d_3dv(u, v, domain, py_mpp.parameters.DGRID_NE)

    if grid_type < 4:
        if nested:           
            def bounded_cubed_sphere_transform(u, v):
                I = slice(Is, Ie)
                J = slice(Js, Je)
                utmp = c2 * (u[I,shift(J,-1)]+u[I,shift(J,2)]) + c1 * (u[I,J]+u[I,shift(J,1)])
                vtmp = c2 * (v[shift(I,-1),J]+v[shift(I,2),J]) + c1 * (v[I,J]+v[shift(I,1),J])
            
                # Transform local a-grid winds into latitude-longitude coordinates
                ua = a11 * utmp + a12 * vtmp
                va = a21 * utmp + a22 * vtmp
                
                ua = xnp.pad(ua, n_halo)
                va = xnp.pad(va, n_halo)
                return ua, va
            
            transform = bounded_cubed_sphere_transform
        else:
            def unbounded_cubed_sphere_transform(u, v):
                utmp = xnp.pad(xnp.zeros((nx_center, ny_center), dtype=default_float), n_halo)
                vtmp = xnp.pad(xnp.zeros((nx_center, ny_center), dtype=default_float), n_halo)

                I = slice(Is+1 if west_edge else Is, Ie-1 if east_edge else Ie)
                J = slice(Js+1 if south_edge else Js, Je-1 if north_edge else Je)
                utmp = update(utmp, at=(I,J), to=c2 * (u[I, shift(J,-1)] + u[I, shift(J,2)]) + c1 * (u[I, J] + u[I, shift(J,1)]))
                vtmp = update(vtmp, at=(I,J), to=c2 * (v[shift(I,-1), J] + v[shift(I,2), J]) + c1 * (v[I, J] + v[shift(I,1), J]))
                
                if south_edge:
                    j = Js
                    I = slice(Is, Ie+1)
                    wv = xnp.pad(v[I, j] * dy[I, j], n_halo)

                    I = slice(Is, Ie)
                    vtmp = update(vtmp, at=(I, j), to=2.0 * (wv[I] + wv[shift(I,1)]) / (dy[I, j] + dy[shift(I,1), j]))
                    utmp = update(utmp, at=(I, j), to=(2.0 * (u[I, j] * dx[I, j] + u[I, j+1] * dx[I, j+1]) /
                                             (dx[I, j] + dx[I, j+1])))

                if north_edge:
                    # Subtract 1 from Je because Je is one-past the end (and we want the last element of the range)
                    j = Je-1
                    I = slice(Is, Ie+1)
                    wv = xnp.pad(v[I, j] * dy[I, j], n_halo)

                    I = slice(Is, Ie)
                    vtmp = update(vtmp, at=(I, j), to=2.0 * (wv[I] + wv[shift(I,1)]) / (dy[I, j] + dy[shift(I,1), j]))
                    utmp = update(utmp, at=(I, j), to=(2.0 * (u[I, j] * dx[I, j] + u[I, j+1] * dx[I, j+1]) /
                                              (dx[I, j] + dx[I, j+1])))

                if west_edge:
                    i = Is
                    J = slice(Js, Je)
                    wv_0 = xnp.pad(v[i, J] * dy[i, J], n_halo)
                    wv_1 = xnp.pad(v[i+1, J] * dy[i+1, J], n_halo)

                    J = slice(Js, Je+1)
                    wu = xnp.pad(u[i, J] * dx[i, J], n_halo)
                    
                    J = slice(Js, Je)
                    utmp = update(utmp, at=(i, J), to=2.0 * (wu[J] + wu[shift(J,1)]) / (dx[i, J] + dx[i, shift(J,1)]))
                    vtmp = update(vtmp, at=(i, J), to=2.0 * (wv_0[J] + wv_1[J]) / (dy[i, J] + dy[i+1, J]))

                if east_edge:
                    # Subtract 1 from Ie because Ie is one-past the end (and we want the last element of the range)
                    i = Ie-1
                    J = slice(Js, Je)
                    wv_0 = xnp.pad((v[i, J] * dy[i, J]), n_halo)
                    wv_1 = xnp.pad((v[i+1, J] * dy[i+1, J]), n_halo)

                    J = slice(Js, Je+1)
                    wu = xnp.pad(u[i, J] * dx[i, J], n_halo)
                    
                    J = slice(Js, Je)
                    utmp = update(utmp, at=(i, J), to=2.0 * (wu[J] + wu[shift(J,1)]) / (dx[i, J] + dx[i, shift(J,1)]))
                    vtmp = update(vtmp, at=(i, J), to=2.0 * (wv_0[J] + wv_1[J]) / (dy[i, J] + dy[i+1, shift(J,1)]))

                # Transform local a-grid winds into latitude-longitude coordinates
                I = slice(Is, Ie)
                J = slice(Js, Je)
                ua = a11 * utmp[I, J] + a12 * vtmp[I, J]
                va = a21 * utmp[I, J] + a22 * vtmp[I, J]
                
                ua = xnp.pad(ua, n_halo)
                va = xnp.pad(va, n_halo)
                return ua, va
            
            transform = unbounded_cubed_sphere_transform
    else:
        def cartesian_geometry_transform(u, v):
            # Simple Cartesian Geometry:
            I = slice(Is, Ie)
            J = slice(Js, Je)
            ua = a2 * (u[I, shift(J,-1)] + u[I, shift(J,2)]) + a1 * (u[I, J] + u[I, shift(J,1)])
            va = a2 * (v[shift(I,-1), J] + v[shift(I,2), J]) + a1 * (v[I, J] + v[shift(I,1), J])
            
            ua = xnp.pad(ua, n_halo)
            va = xnp.pad(va, n_halo)
            return ua, va
        
        transform = cartesian_geometry_transform

    # Map the function `transform` over axis 2 of u and v stacking the results along axis 2 of the outputs.
    # In effect `transform` is the kernel of a for-loop, where the loop index is for axis 2 of each array.
    vtransform = vmap(transform, 2, 2)
    ua, va = vtransform(u, v)
    
    return ua, va