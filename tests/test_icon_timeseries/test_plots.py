"""Tests for plots."""
# Standard library
import glob

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import xarray as xr

# First-party
from icon_timeseries.plot import deagg_sum
from icon_timeseries.plot import deaverage
from icon_timeseries.plot import plot_ts
from icon_timeseries.plot import prepare_data

gridfile = "/store/s83/tsm/ICON_INPUT/icon-1e_dev/ICON-1E_DOM01.nc"
exp = "/store/s83/osm/ICON-CH1-EPS/FCST_RING/22111600/lfff000[0-3]0000"

# def test_plot_domain():
#     """Test the quicklook for a masked domain."""
#     gd = get_grid(gridfile)
#     # get the domain info
#     dom_pts, dom_name = get_domain("ch")
#     # mask the grid
#     gd.mask_domain(dom_pts)
#     # plot domain
#     fig = plot_domain(gd, dom_name)
#     assert fig.get_axes()[0].get_title() == "domain Swiss domain"


def test_ts():
    """Test plotting one time series."""
    domain = "ch"
    filelist = glob.glob(exp)
    da_mean, _ = prepare_data(
        filelist,
        "T",
        80,
        gridfile,
        domain=domain,
        deagg="no",
    )
    plot_ts(da_mean.values, da_mean.valid_time.values)
    np.testing.assert_array_equal(
        plt.gca().lines[0].get_xydata()[:, 1],
        da_mean.values,
        "y values in plot do not match input values",
    )


def _create_test_da() -> xr.DataArray:
    """Create simple sample DataArray"""
    smpl_data = np.random.rand(5, 5)
    t = np.arange("2022-11-28T00:00", "2022-11-28T05:00", dtype="datetime64[h]")
    x = np.arange(0, 5, dtype="int32")
    smpl_coords = (t, x)
    smpl_da = xr.DataArray(smpl_data, dims=["valid_time", "x"], coords=smpl_coords)
    return smpl_da


def test_deaverage():
    """Test deaveraging over valid_time."""
    test_da = _create_test_da()
    av_da = test_da.copy()
    for i in range(2, 5):
        av_da[i, :] = (float(i - 1) * av_da[i - 1, :] + test_da[i, :]) / float(i)
    deav_da = deaverage(av_da)
    np.testing.assert_array_almost_equal(deav_da, test_da, decimal=14)


def test_deagg_sum():
    """Test deaggregation of sums over valid_time."""
    test_da = _create_test_da()
    agg_da = test_da.copy()
    for i in range(2, 5):
        agg_da[i, :] = agg_da[i - 1, :] + test_da[i]
    deagg_da = deagg_sum(agg_da)
    np.testing.assert_array_almost_equal(deagg_da, test_da, decimal=14)
