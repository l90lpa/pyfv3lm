from enum import Enum
from dataclasses import dataclass, field
from nptyping import NDArray, Int, Shape, Float, Float64, Bool
from dataclasses_json import dataclass_json, config
from .json_utils import ndarray_encoder, ndarray_decoder
import chex

class AxisType(Enum):
    CENTER = 1
    LEFT   = 2
    RIGHT  = 3
    INNER  = 4
    OUTER  = 5


@dataclass_json
@dataclass(frozen=True)
class LocalGridSpace:
    _center_data_shape: tuple[int, int]
    _center_domain_origin: tuple[int, int]
    _center_domain_shape: tuple[int, int]
    _center_compute_origin: tuple[int, int]
    _center_compute_shape: tuple[int, int]
    south_edge: bool   # Indicate whether the sub tile is adjacent to the south edge of embedding tile
    north_edge: bool   # Indicate whether the sub tile is adjacent to the north edge of embedding tile
    west_edge: bool   # Indicate whether the sub tile is adjacent to the west edge of embedding tile
    east_edge: bool   # Indicate whether the sub tile is adjacent to the east edge of embedding tile

    def _get_shape(self, center_shape: tuple[int, int], x_axis: AxisType, y_axis: AxisType):
        nx = center_shape[0]
        if x_axis == AxisType.INNER:
            nx -= 1
        elif x_axis == AxisType.OUTER:
            nx += 1

        ny = center_shape[1]
        if y_axis == AxisType.INNER:
            ny -= 1
        elif y_axis == AxisType.OUTER:
            ny += 1

        return nx, ny

    def data_shape(self, x_axis: AxisType, y_axis: AxisType):
        return self._get_shape(self._center_data_shape, x_axis, y_axis)
    
    def compute_shape(self, x_axis: AxisType, y_axis: AxisType):
        return self._get_shape(self._center_compute_shape, x_axis, y_axis)
    
    def domain_shape(self, x_axis: AxisType, y_axis: AxisType):
        return self._get_shape(self._center_domain_shape, x_axis, y_axis)
    
    def Isc(self):
        return self._center_compute_origin[0]

    def Iec(self):
        return self._center_compute_origin[0] + self._center_compute_shape[0]
    
    def Is(self):
        return self._center_domain_origin[0]

    def Ie(self):
        return self._center_domain_origin[0] + self._center_domain_shape[0]

    def Jsc(self):
        return self._center_compute_origin[1]

    def Jec(self):
        return self._center_compute_origin[1] + self._center_compute_shape[1]
    
    def Js(self):
        return self._center_domain_origin[1]

    def Je(self):
        return self._center_domain_origin[1] + self._center_domain_shape[1]


@dataclass(frozen=True)
class GridBounds:
    Is : int
    Ie : int
    Js : int
    Je : int
    Isd: int
    Ied: int
    Jsd: int
    Jed: int
    Isc: int
    Iec: int
    Jsc: int
    Jec: int


def to_local_grid_space(bounds: GridBounds, npx, npy, npz):
    assert bounds.Isd <= bounds.Is  and bounds.Ie  <= bounds.Ied
    assert bounds.Isd <= bounds.Isc and bounds.Iec <= bounds.Ied
    assert bounds.Jsd <= bounds.Js  and bounds.Je  <= bounds.Jed
    assert bounds.Jsd <= bounds.Jsc and bounds.Jec <= bounds.Jed

    data_shape = ((bounds.Ied - bounds.Isd) + 1, (bounds.Jed - bounds.Jsd) + 1)
    domain_origin = (bounds.Is - bounds.Isd, bounds.Js - bounds.Jsd)
    domain_shape = ((bounds.Ie - bounds.Is) + 1, (bounds.Je - bounds.Js) + 1)
    compute_origin = (bounds.Isc - bounds.Isd, bounds.Jsc - bounds.Jsd)
    compute_shape = ((bounds.Iec - bounds.Isc) + 1, (bounds.Jec - bounds.Jsc) + 1)
    south_edge = bounds.Js == 1
    north_edge = bounds.Je == (npy - 1)
    west_edge = bounds.Is == 1
    east_edge = bounds.Ie == (npx - 1)

    return LocalGridSpace(
        data_shape,
        domain_origin,
        domain_shape,
        compute_origin,
        compute_shape,
        south_edge,
        north_edge,
        west_edge,
        east_edge
        
    )

@dataclass_json
@chex.dataclass
# @dataclass(frozen=True)
class Grid:
    grid_64: NDArray[Shape["*,*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    agrid_64: NDArray[Shape["*,*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    area_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    area_c_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    sina_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    cosa_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dx_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dy_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dxc_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dyc_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dxa_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dya_64: NDArray[Shape["*,*"], Float64] | None = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

    grid: NDArray[Shape["*,*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    agrid: NDArray[Shape["*,*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    area: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    area_c: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rarea: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rarea_c: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

    sina: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    cosa: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    e1: NDArray[Shape["*,*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    e2: NDArray[Shape["*,*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dx: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dy: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dxc: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dyc: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dxa: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    dya: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rdx: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rdy: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rdxc: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rdyc: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rdxa: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rdya: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

# Scalars:
    edge_s: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    edge_n: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    edge_w: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    edge_e: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

# Vector:
    edge_vect_s: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    edge_vect_n: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    edge_vect_w: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    edge_vect_e: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

# Scalar:
    ex_s: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ex_n: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ex_w: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ex_e: NDArray[Shape["*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

    l2c_u: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    l2c_v: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

# Divergence damping:
    divg_u: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    divg_v: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
# Del6 diffusion:
    del6_u: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    del6_v: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
# Cubed_2_latlon:
    a11: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    a12: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    a21: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    a22: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
# Latlon_2_cubed:
    z11: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    z12: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    z21: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    z22: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

    # Intentionally commented out
    # w00: NDArray[Shape["*,*"], Float]

    cosa_u: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    cosa_v: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    cosa_s: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    sina_u: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    sina_v: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rsin_u: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rsin_v: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rsina : NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    rsin2 : NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    
    ee1   : NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ee2   : NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ec1   : NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ec2   : NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ew    : NDArray[Shape["*,*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    es    : NDArray[Shape["*,*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))


# -- 3D Super grid to contain all geometrical factors --
    
    # The 3rd dimension is 9
    sin_sg: NDArray[Shape["*,*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    cos_sg: NDArray[Shape["*,*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

# ------------------------------------------------------

# Unit Normal vectors at cell edges:
    en1: NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    en2: NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

# Extended Cubed cross-edge winds:
    eww: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    ess: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

# Unit vectors for lat-lon grid:
    vlon: NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    vlat: NDArray[Shape["*,*,*"], Float64] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    fc: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    f0: NDArray[Shape["*,*"], Float] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))

    iinta: NDArray[Shape["*,*,*"], Int] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    jinta: NDArray[Shape["*,*,*"], Int] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    iintb: NDArray[Shape["*,*,*"], Int] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
    jintb: NDArray[Shape["*,*,*"], Int] = field(metadata=config(decoder=ndarray_decoder, encoder=ndarray_encoder))
  
# Scalar data
    
    # global domain
    npx_g: Int
    npy_g: Int
    ntiles_g: Int

    global_area: Float64
    # g_sum_initialized: Bool = False # Not currently used but can be useful
    sw_corner: Bool
    se_corner: Bool
    ne_corner: Bool
    nw_corner: Bool

    da_min: Float64
    da_max: Float64
    da_min_c: Float64
    da_max_c: Float64

    acapn: Float
    acaps: Float
    globalarea: Float  # Total global area
     
    latlon: Bool
    cubed_sphere: Bool
    have_south_pole: Bool
    have_north_pole: Bool
    stretched_grid: Bool
    square_domain: Bool

# Convenience pointers
    grid_type: Int
    nested: Bool


def shallow_copy_grid(src_grid):
    grid_copy = Grid(
        grid_64 = None,
        agrid_64 = None,
        area_64 = None,
        area_c_64 = None,
        sina_64 = None,
        cosa_64 = None,
        dx_64 = None,
        dy_64 = None,
        dxc_64 = None,
        dyc_64 = None,
        dxa_64 = None,
        dya_64 = None,
        grid = src_grid.grid,
        agrid = src_grid.agrid,
        area = src_grid.area,
        area_c = src_grid.area_c,
        rarea = src_grid.rarea,
        rarea_c = src_grid.rarea_c,
        sina = src_grid.sina,
        cosa = src_grid.cosa,
        e1 = src_grid.e1,
        e2 = src_grid.e2,
        dx = src_grid.dx,
        dy = src_grid.dy,
        dxc = src_grid.dxc,
        dyc = src_grid.dyc,
        dxa = src_grid.dxa,
        dya = src_grid.dya,
        rdx = src_grid.rdx,
        rdy = src_grid.rdy,
        rdxc = src_grid.rdxc,
        rdyc = src_grid.rdyc,
        rdxa = src_grid.rdxa,
        rdya = src_grid.rdya,
        edge_s = src_grid.edge_s,
        edge_n = src_grid.edge_n,
        edge_w = src_grid.edge_w,
        edge_e = src_grid.edge_e,
        edge_vect_s = src_grid.edge_vect_s,
        edge_vect_n = src_grid.edge_vect_n,
        edge_vect_w = src_grid.edge_vect_w,
        edge_vect_e = src_grid.edge_vect_e,
        ex_s = src_grid.ex_s,
        ex_n = src_grid.ex_n,
        ex_w = src_grid.ex_w,
        ex_e = src_grid.ex_e,
        l2c_u = src_grid.l2c_u,
        l2c_v = src_grid.l2c_v,
        divg_u = src_grid.divg_u,
        divg_v = src_grid.divg_v,
        del6_u = src_grid.del6_u,
        del6_v = src_grid.del6_v,
        a11 = src_grid.a11,
        a12 = src_grid.a12,
        a21 = src_grid.a21,
        a22 = src_grid.a22,
        z11 = src_grid.z11,
        z12 = src_grid.z12,
        z21 = src_grid.z21,
        z22 = src_grid.z22,
        cosa_u = src_grid.cosa_u,
        cosa_v = src_grid.cosa_v,
        cosa_s = src_grid.cosa_s,
        sina_u = src_grid.sina_u,
        sina_v = src_grid.sina_v,
        rsin_u = src_grid.rsin_u,
        rsin_v = src_grid.rsin_v,
        rsina = src_grid.rsina,
        rsin2 = src_grid.rsin2,
        ee1 = src_grid.ee1,
        ee2 = src_grid.ee2,
        ec1 = src_grid.ec1,
        ec2 = src_grid.ec2,
        ew = src_grid.ew,
        es = src_grid.es,
        sin_sg = src_grid.sin_sg,
        cos_sg = src_grid.cos_sg,
        en1 = src_grid.en1,
        en2 = src_grid.en2,
        eww = src_grid.eww,
        ess = src_grid.ess,
        vlon = src_grid.vlon,
        vlat = src_grid.vlat,
        fc = src_grid.fc,
        f0 = src_grid.f0,
        iinta = src_grid.iinta,
        jinta = src_grid.jinta,
        iintb = src_grid.iintb,
        jintb = src_grid.jintb,
        npx_g = src_grid.npx_g,
        npy_g = src_grid.npy_g,
        ntiles_g = src_grid.ntiles_g,
        global_area = src_grid.global_area,
        sw_corner = src_grid.sw_corner,
        se_corner = src_grid.se_corner,
        ne_corner = src_grid.ne_corner,
        nw_corner = src_grid.nw_corner,
        da_min = src_grid.da_min,
        da_max = src_grid.da_max,
        da_min_c = src_grid.da_min_c,
        da_max_c = src_grid.da_max_c,
        acapn = src_grid.acapn,
        acaps = src_grid.acaps,
        globalarea = src_grid.globalarea,
        latlon = src_grid.latlon,
        cubed_sphere = src_grid.cubed_sphere,
        have_south_pole = src_grid.have_south_pole,
        have_north_pole = src_grid.have_north_pole,
        stretched_grid = src_grid.stretched_grid,
        square_domain = src_grid.square_domain,
        grid_type = src_grid.grid_type,
        nested = src_grid.nested
    )
    return grid_copy