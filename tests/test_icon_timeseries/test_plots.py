"""Tests for plots."""
# Standard library
import glob

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

# First-party
from icon_timeseries.plot import plot_ts
from icon_timeseries.prepare_data import prepare_meanmax

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
    da_mean, _ = prepare_meanmax(
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
