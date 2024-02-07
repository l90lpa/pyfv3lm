from functools import partial

from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.interpreters.mlir import token_type, custom_call
from jax.lib import xla_client

import numpy as np

from . import jax_mpp_ops
from .effect import ordered_effect
from .hashable import pack_hashable_array, unpack_hashable_array

# print(jax_mpp_ops.registrations())

##############################################################################
## Helper functions

def get_default_layouts(operands, order="c"):
    (token,) = token_type()
    layouts = []

    if order == "c":
        default_layout = lambda t: tuple(range(len(t.shape) - 1, -1, -1))
    elif order == "f":
        default_layout = lambda t: tuple(range(len(t.shape)))
    else:
        raise ValueError(f"Unknown order: {order}")

    for op in operands:
        if isinstance(op, (mlir.ir.Value)):
            if op.type == token:
                layouts.append(())
            else:
                tensor_type = mlir.ir.RankedTensorType(op.type)
                layouts.append(default_layout(tensor_type))

        elif isinstance(op, mlir.ir.RankedTensorType):
            layouts.append(default_layout(op))

        elif op == token:
            layouts.append(())

        else:
            raise ValueError(f"Unknown operand type: {type(op)}")

    return layouts


##############################################################################
## Register jax_mpp_ops with XLA


for _name, _value in jax_mpp_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")


##############################################################################
## Define JAX primitives for jax_mpp_ops
    

##############################################################################
## Define define_layout2d primitives for jax_mpp_ops

def py_mpp_define_layout2d(global_indices, ndivs, layout):
    return _define_layout2d_prim.bind(global_indices, ndivs, layout)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _define_layout2d_abstract(global_indices, ndivs, layout):
    return (ShapedArray(global_indices.shape, global_indices.dtype), ShapedArray(layout.shape, layout.dtype))


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _define_layout2d_lowering(ctx, global_indices, ndivs, layout):
    
    # Determine the input types from the abstract values
    global_indices_aval, _, layout_aval = ctx.avals_in
    global_indices_nptype = global_indices_aval.dtype
    layout_nptype = layout_aval.dtype

    global_indices_type = mlir.ir.RankedTensorType(global_indices.type)
    global_indices_shape = global_indices_type.shape
    n_indices = np.prod(global_indices_shape).astype(np.int32)

    layout_type = mlir.ir.RankedTensorType(layout.type)
    layout_shape = layout_type.shape
    n_layout = np.prod(layout_shape).astype(np.int32)

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands=[
        global_indices,
        mlir.ir_constant(n_indices),
        ndivs,
        layout,
        mlir.ir_constant(n_layout),
        token
    ]
    
    result_types=[
        global_indices_type,
        layout_type,
        token_type()
    ]

    # We dispatch a different call depending on the dtype
    if np.dtype(global_indices_nptype) == np.int32:
        op_name = "jax_py_mpp_define_layout2D"
    else:
        raise NotImplementedError(f"Unsupported dtype {np.dtype(global_indices_nptype)}")

    results_obj = custom_call(
        op_name,
        result_types=result_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands, order='f'),
        result_layouts=get_default_layouts(result_types, order='f'),
        has_side_effect=True
    )

    results = list(results_obj.results)
    token = results.pop(-1)
    ctx.set_token_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results





_define_layout2d_prim = core.Primitive("py_mpp_define_layout2d")
_define_layout2d_prim.multiple_results = True
_define_layout2d_prim.def_impl(partial(xla.apply_primitive, _define_layout2d_prim))
_define_layout2d_prim.def_abstract_eval(_define_layout2d_abstract)
# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_define_layout2d_prim, _define_layout2d_lowering, platform='cpu')


##############################################################################
## Define update_domain2d_3dv_ad primitives for jax_mpp_ops

def py_mpp_update_domain2d_3dv_ad(fieldx, fieldy, domain_handle, grid_type):
    domain_handle = pack_hashable_array(domain_handle)
    return _update_domain2d_3dv_ad_prim.bind(fieldx, fieldy, domain_handle, grid_type)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _update_domain2d_3dv_ad_abstract(fieldx, fieldy, domain_handle, grid_type):
    domain_handle = unpack_hashable_array(domain_handle)
    return (ShapedArray(fieldx.shape, fieldx.dtype), ShapedArray(fieldy.shape, fieldy.dtype))


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _update_domain2d_3dv_ad_lowering(ctx, fieldx, fieldy, domain_handle, grid_type):
    domain_handle = unpack_hashable_array(domain_handle)
    
    # Determine the input types from the abstract values
    fieldx_aval, fieldy_aval, *_ = ctx.avals_in
    fieldx_nptype = fieldx_aval.dtype
    fieldy_nptype = fieldy_aval.dtype

    fieldx_type = mlir.ir.RankedTensorType(fieldx.type)
    fieldx_shape = fieldx_type.shape
    (xi, xj, xk) = fieldx_shape
    # n_fieldx = np.prod(fieldx_shape).astype(np.float64)

    fieldy_type = mlir.ir.RankedTensorType(fieldy.type)
    fieldy_shape = fieldy_type.shape
    (yi, yj, yk) = fieldy_shape
    # n_fieldy = np.prod(fieldy_shape).astype(np.float64)

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands=[
                fieldx,
                mlir.ir_constant(xi),
                mlir.ir_constant(xj),
                mlir.ir_constant(xk),
                fieldy,
                mlir.ir_constant(yi),
                mlir.ir_constant(yj),
                mlir.ir_constant(yk),
                domain_handle,
                grid_type,
                token
            ]
    
    result_types=[fieldx_type, fieldy_type, token_type()]

    # We dispatch a different call depending on the dtype
    if np.dtype(fieldx_nptype) != np.dtype(fieldy_nptype):
        raise ValueError(f"Both field types must be the same, got fieldx type = {np.dtype(fieldx_nptype)} and fieldy type = {np.dtype(fieldy_nptype)}.") 
    if not (len(fieldx_shape) == 3 and len(fieldy_shape) == 3):
        raise ValueError(f"Both fields must be 3 dimensional, got fieldx dims = {len(fieldx_shape)} and fieldy dims = {len(fieldy_shape)}.") 
    dtype = np.dtype(fieldx_nptype)
    if dtype == np.float32:
        op_name = "jax_py_mpp_update_domain2d_r4_3dv_ad"
    elif dtype == np.float64:
        op_name = "jax_py_mpp_update_domain2d_r8_3dv_ad"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}.")

    result_obj = custom_call(
        op_name,
        result_types=result_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands, order='f'),
        result_layouts=get_default_layouts(result_types, order='f'),
        has_side_effect=True
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results


_update_domain2d_3dv_ad_prim = core.Primitive("py_mpp_update_domain2d_3dv_ad")
_update_domain2d_3dv_ad_prim.multiple_results = True
_update_domain2d_3dv_ad_prim.def_impl(partial(xla.apply_primitive, _update_domain2d_3dv_ad_prim))
_update_domain2d_3dv_ad_prim.def_abstract_eval(_update_domain2d_3dv_ad_abstract)
# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_update_domain2d_3dv_ad_prim, _update_domain2d_3dv_ad_lowering, platform='cpu')


##############################################################################
## Define update_domain2d_3dv primitives for jax_mpp_ops

def py_mpp_update_domain2d_3dv(fieldx, fieldy, domain_handle, grid_type):
    domain_handle = pack_hashable_array(domain_handle)
    return _update_domain2d_3dv_prim.bind(fieldx, fieldy, domain_handle, grid_type)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _update_domain2d_3dv_abstract(fieldx, fieldy, domain_handle, grid_type):
    domain_handle = unpack_hashable_array(domain_handle)
    return (ShapedArray(fieldx.shape, fieldx.dtype), ShapedArray(fieldy.shape, fieldy.dtype))


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _update_domain2d_3dv_lowering(ctx, fieldx, fieldy, domain_handle, grid_type):
    domain_handle = unpack_hashable_array(domain_handle)
    
    # Determine the input types from the abstract values
    fieldx_aval, fieldy_aval, *_ = ctx.avals_in
    fieldx_nptype = fieldx_aval.dtype
    fieldy_nptype = fieldy_aval.dtype

    fieldx_type = mlir.ir.RankedTensorType(fieldx.type)
    fieldx_shape = fieldx_type.shape
    (xi, xj, xk) = fieldx_shape
    # n_fieldx = np.prod(fieldx_shape).astype(np.float64)

    fieldy_type = mlir.ir.RankedTensorType(fieldy.type)
    fieldy_shape = fieldy_type.shape
    (yi, yj, yk) = fieldy_shape
    # n_fieldy = np.prod(fieldy_shape).astype(np.float64)

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands=[
                fieldx,
                mlir.ir_constant(xi),
                mlir.ir_constant(xj),
                mlir.ir_constant(xk),
                fieldy,
                mlir.ir_constant(yi),
                mlir.ir_constant(yj),
                mlir.ir_constant(yk),
                domain_handle,
                grid_type,
                token
            ]
    
    result_types=[fieldx_type, fieldy_type, token_type()]

    # We dispatch a different call depending on the dtype
    if np.dtype(fieldx_nptype) != np.dtype(fieldy_nptype):
        raise ValueError(f"Both field types must be the same, got fieldx type = {np.dtype(fieldx_nptype)} and fieldy type = {np.dtype(fieldy_nptype)}.") 
    if not (len(fieldx_shape) == 3 and len(fieldy_shape) == 3):
        raise ValueError(f"Both fields must be 3 dimensional, got fieldx dims = {len(fieldx_shape)} and fieldy dims = {len(fieldy_shape)}.") 
    dtype = np.dtype(fieldx_nptype)
    if dtype == np.float32:
        op_name = "jax_py_mpp_update_domain2d_r4_3dv"
    elif dtype == np.float64:
        op_name = "jax_py_mpp_update_domain2d_r8_3dv"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}.")

    result_obj = custom_call(
        op_name,
        result_types=result_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands, order='f'),
        result_layouts=get_default_layouts(result_types, order='f'),
        has_side_effect=True
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results


def update_domain2d_3dv_value_and_jvp(
    primal_args,
    tangent_args,
    domain_handle,
    grid_type,
):
    fieldx, fieldy = primal_args
    fieldx_t, fieldy_t = tangent_args

    val = _update_domain2d_3dv_prim.bind(fieldx, fieldy, domain_handle, grid_type)
    jvp = _update_domain2d_3dv_prim.bind(fieldx_t, fieldy_t, domain_handle, grid_type)

    return val, jvp


def update_domain2d_3dv_transpose_rule(
    cotangent_args, *primal_args, domain_handle, grid_type
):
    fieldx_ct, fieldy_ct = cotangent_args
    res = _update_domain2d_3dv_ad_prim.bind(fieldx_ct, fieldy_ct, domain_handle, grid_type)
    return res



_update_domain2d_3dv_prim = core.Primitive("py_mpp_update_domain2d_3dv")
_update_domain2d_3dv_prim.multiple_results = True
_update_domain2d_3dv_prim.def_impl(partial(xla.apply_primitive, _update_domain2d_3dv_prim))
_update_domain2d_3dv_prim.def_abstract_eval(_update_domain2d_3dv_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_update_domain2d_3dv_prim, _update_domain2d_3dv_lowering, platform='cpu')


ad.primitive_jvps[_update_domain2d_3dv_prim] = update_domain2d_3dv_value_and_jvp
ad.primitive_transposes[_update_domain2d_3dv_prim] = update_domain2d_3dv_transpose_rule