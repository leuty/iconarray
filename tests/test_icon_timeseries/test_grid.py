"""Tests for ICON grid reading."""
# First-party
from icon_timeseries.handle_grid import get_grid
from icon_timeseries.read_grib import var_from_files

gridfile = "/store/s83/tsm/ICON_INPUT/icon-1e_dev/ICON-1E_DOM01.nc"
wrong_gridfile = (
    "/store/s83/osm/ICON-GRIDS/icon_grid_0001_R19B08_mch/"
    "icon_grid_0001_R19B08_mch_DOM01.nc"
)
gribfile = "/store/s83/osm/ICON-CH1-EPS/FCST_RING/22111600/lfff00000000"


def test_get_grid():
    """Test reading the grid file and creating a triangulation object."""
    gd = get_grid(gridfile)
    assert gd.edges.shape == (1543753, 2), "gd should contain 1543753 edges"
    assert gd.triangles.shape == (1028172, 3), "gd should contain 1028172 cells"
    assert gd.x.shape == (515582,), "gd should contain 515582 vertices"
    assert gd.cx.shape == (1028172,), "gd should contain 1028172 cell centers"


def test_grid_compatibility():
    """Test if grid information matches the data."""
    da = var_from_files(
        [gribfile],
        "T",
    )
    gd = get_grid(gridfile)
    assert gd.check_compatibility(da) is True, "grid and data should match in size"
    gd = get_grid(wrong_gridfile)
    assert gd.check_compatibility(da) is False, "grid and data should not match in size"


def test_mask_grid():
    """Test the mask generation for the Swiss domain."""
    gd = get_grid(gridfile)
    gd.mask_domain()
    # pylint: disable=invalid-unary-operand-type
    assert sum(~gd.mask) == 95156, "the Swiss domain should contain 95156 points"
    # pylint: enable=invalid-unary-operand-type
