from .numpy_mpp_ops import mpp_ops

import numpy as np


def py_mpp_init():
    mpp_ops.py_mpp_init()


def py_mpp_pe():
    pe = np.array([-1], dtype=np.int32)
    mpp_ops.py_mpp_pe(pe)
    return pe[0]


def py_mpp_npes():
    npes = np.array([-1], dtype=np.int32)
    mpp_ops.py_mpp_npes(npes)
    return npes[0]


def py_mpp_set_root_pe(root):
    mpp_ops.py_mpp_set_root_pe(root)


def py_mpp_exit():
    mpp_ops.py_mpp_exit()


def py_mpp_sync():
    mpp_ops.py_mpp_sync()


def py_mpp_domains_init():
    mpp_ops.py_mpp_domains_init()


def py_mpp_domains_set_stack_size(stack_size):
    mpp_ops.py_mpp_domains_set_stack_size(stack_size)


def py_mpp_domains_exit():
    mpp_ops.py_mpp_domains_exit()


def py_mpp_allocate_domain2d():
    domain_handle = np.array([0,0], dtype=np.int32)
    mpp_ops.py_mpp_allocate_domain2d(domain_handle)
    return domain_handle

    
def py_mpp_deallocate_domain2d(domain_handle):
    mpp_ops.py_mpp_allocate_domain2d(domain_handle)


def py_mpp_get_compute_domain2d(domain_handle):
    compute_domain_indices = np.array([-1,-1,-1,-1], dtype=np.int32)
    mpp_ops.py_mpp_get_compute_domain2d(domain_handle, compute_domain_indices)
    return compute_domain_indices



def py_mpp_get_data_domain2d(domain_handle):
    data_domain_indices = np.array([-1,-1,-1,-1], dtype=np.int32)
    mpp_ops.py_mpp_get_data_domain2d(domain_handle, data_domain_indices)
    return data_domain_indices



def py_mpp_define_layout2d(global_indices, ndivs):
    layout = np.array([-1,-1], dtype=np.int32)
    mpp_ops.py_mpp_define_layout2d(global_indices, ndivs, layout)
    return layout


def py_mpp_define_domains2d(global_indices, layout, domain_handle,
                            xhalo, yhalo, print_info=0):
    mpp_ops.py_mpp_define_domains2d(global_indices, layout, domain_handle,
                                    xhalo, yhalo, print_info)


def py_mpp_update_domain2d_3dv(fieldx, fieldy, domain_handle, grid_type):
    assert fieldx.dtype == fieldy.dtype
    dtype = fieldx.dtype
    if dtype == np.float32:
        mpp_ops.py_mpp_update_domain2d_r4_3dv(fieldx, fieldy, domain_handle, grid_type)
    elif dtype == np.float64:
        mpp_ops.py_mpp_update_domain2d_r8_3dv(fieldx, fieldy, domain_handle, grid_type)
    else:
        ValueError(f"Unsupported dtype {dtype}.")
    return fieldx, fieldy


def py_mpp_update_domain2d_3dv_ad(fieldx, fieldy, domain_handle, grid_type):
    assert fieldx.dtype == fieldy.dtype
    dtype = fieldx.dtype
    if dtype == np.float32:
        mpp_ops.py_mpp_update_domain2d_r4_3dv_ad(fieldx, fieldy, domain_handle, grid_type)
    elif dtype == np.float64:
        mpp_ops.py_mpp_update_domain2d_r8_3dv_ad(fieldx, fieldy, domain_handle, grid_type)
    else:
        ValueError(f"Unsupported dtype {dtype}.")
    return fieldx, fieldy

