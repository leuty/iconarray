"""Tests for plots."""
# Standard library
import glob

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

# First-party
from icon_timeseries.plot import plot_histograms
from icon_timeseries.plot import plot_ts
from icon_timeseries.plot import plot_ts_multiple
from icon_timeseries.prepare_data import prepare_masked_da
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
        deagg=False,
    )
    plot_ts(da_mean.values, da_mean.valid_time.values)
    np.testing.assert_array_equal(
        plt.gca().lines[0].get_xydata()[:, 1],
        da_mean.values,
        "y values in plot do not match input values",
    )


def test_ts_multi():
    """ "Test plotting time series of min and max."""
    domain = "ch"
    filelist = glob.glob(exp)
    da_mean, da_max = prepare_meanmax(
        filelist,
        "T",
        80,
        gridfile,
        domain=domain,
        deagg=False,
    )
    da_dict = {
        "mean": {"exp1": da_mean},
        "max": {"exp1": da_max},
    }
    plot_ts_multiple(da_dict, save=False)
    np.testing.assert_array_equal(
        plt.gcf().get_axes()[0].lines[0].get_xydata()[:, 1],
        da_mean.values,
        "y values in plot do not match input values for mean",
    )
    np.testing.assert_array_equal(
        plt.gcf().get_axes()[1].lines[0].get_xydata()[:, 1],
        da_max.values,
        "y values in plot do not match input values for max",
    )


def test_hist():
    """ "Test plotting one histogram."""
    domain = "ch"
    filelist = glob.glob(exp)
    min_bin = 250
    max_bin = 290
    nbins = 50
    da = prepare_masked_da(
        filelist,
        "T",
        80,
        gridfile,
        domain=domain,
        deagg=False,
    )
    da_dict = {"exp1": da}
    counts, _ = np.histogram(da.values.flatten(), np.linspace(min_bin, max_bin, nbins))
    plot_histograms(da_dict, min_bin=min_bin, max_bin=max_bin, nbins=nbins, save=False)
    np.testing.assert_array_equal(
        plt.gca().containers[0].datavalues,
        counts,
        "bar values in plot do not match input values",
    )
