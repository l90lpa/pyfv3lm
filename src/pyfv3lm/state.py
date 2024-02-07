from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class State:
    u: jnp.ndarray
    v: jnp.ndarray
    w: jnp.ndarray
    pt: jnp.ndarray
    delp: jnp.ndarray
    q: jnp.ndarray
    delz: jnp.ndarray
    ze0: jnp.ndarray
    ps: jnp.ndarray
    pe: jnp.ndarray
    pk: jnp.ndarray
    peln: jnp.ndarray
    pkz: jnp.ndarray
    q_con: jnp.ndarray
    omga: jnp.ndarray
    phis: jnp.ndarray
    uc: jnp.ndarray
    vc: jnp.ndarray
    ua: jnp.ndarray
    va: jnp.ndarray
    mfx: jnp.ndarray
    mfy: jnp.ndarray
    cx: jnp.ndarray
    cy: jnp.ndarray


@dataclass(frozen=True)
class Parameters:
    bdt: float
    consv_te: float
    kappa: float
    cp_air: float
    zvir: float
    ptop: float
    time_total: float
    npx: int
    npy: int
    npz: int
    nq_tot: int
    ng: int
    ks: int
    ncnst: int
    n_split: int
    q_split: int
    fill: bool
    reproduce_sum: bool
    hydrostatic: bool
    hybrid_z: bool