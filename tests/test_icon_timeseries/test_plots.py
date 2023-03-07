"""Tests for plots."""
# Standard library
import glob
import os

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

# First-party
from icon_timeseries.plot import plot_ts
from icon_timeseries.prepare_data import prepare_meanmax

test_dir = "/store/s83/cmerker/test_data/icon_timeseries/data/test_plots"

gridfile = os.path.join(test_dir, "ICON-1E_DOM01.nc")
exp = os.path.join(test_dir, "lfff000[0-3]0000")

# saved values, reference results
test_mean = np.array([279.10766602, 279.22027588, 279.02398682, 278.8706665])

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


def test_ts_plot():
    """Test plotting one time series."""
    domain = "ch"
    filelist = glob.glob(exp)
    da_mean, _ = prepare_meanmax(
        filelist,
        "T",
        80,
        gridfile,
        domain=domain,
        deagg=False,
    )
    plot_ts(da_mean.values, da_mean.valid_time.values)
    np.testing.assert_array_equal(
        plt.gca().lines[0].get_xydata()[:, 1],
        da_mean.values,
        "y values in plot do not match input values",
    )


def test_ts_results():
    """Test plotting one time series."""
    domain = "ch"
    filelist = glob.glob(exp)
    da_mean, _ = prepare_meanmax(
        filelist,
        "T",
        80,
        gridfile,
        domain=domain,
        deagg=False,
    )
    plot_ts(da_mean.values, da_mean.valid_time.values)
    np.testing.assert_allclose(
        plt.gca().lines[0].get_xydata()[:, 1],
        test_mean,
        rtol=1e-10,
        atol=1e-10,
        err_msg="values in plot do not match saved values",
    )
