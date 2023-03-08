"""Tests for plots."""
# Standard library
import glob
import os

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

# First-party
from icon_timeseries.plot import plot_histograms
from icon_timeseries.plot import plot_ts
from icon_timeseries.plot import plot_ts_multiple
from icon_timeseries.prepare_data import prepare_masked_da
from icon_timeseries.prepare_data import prepare_meanmax

test_single = "/store/s83/cmerker/test_data/icon_timeseries/data/test_plots"
gridfile = os.path.join(test_single, "ICON-1E_DOM01.nc")
exp_s = os.path.join(test_single, "lfff000[0-3]0000")

test_exps = "/store/s83/cmerker/test_data/icon_timeseries/data/test_ens"
exp_1 = os.path.join(test_exps, "ICON-CH1-EPS", "004", "lfff000[0-4]0000")
exp_2 = os.path.join(test_exps, "ICON-CH2-EPS", "004", "lfff000[0-4]0000")

# saved values, reference results
test_mean = np.array([279.10767, 279.22028, 279.024, 278.87067])
test_max = np.array([288.37396, 288.02383, 287.89978, 287.8776])
test_bar = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        3,
        19,
        34,
        62,
        93,
        166,
        324,
        544,
        986,
        1576,
        2627,
        4939,
        7761,
        10301,
        12742,
        13945,
        13329,
        11594,
        11668,
        12453,
        13338,
        15414,
        18417,
        31746,
        50570,
        47403,
        28070,
        24560,
        27610,
        14314,
        2848,
        877,
        265,
        22,
        1,
        0,
    ]
)
test_bar_1 = np.array(
    [
        5109178,
        13554,
        4263,
        2121,
        993,
        483,
        292,
        189,
        121,
        94,
        75,
        41,
        17,
        5,
        4,
        4,
        1,
        2,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)
test_bar_2 = np.array(
    [
        1278832,
        4676,
        1385,
        726,
        382,
        228,
        129,
        77,
        76,
        47,
        27,
        36,
        21,
        8,
        11,
        4,
        5,
        5,
        3,
        3,
        1,
        1,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)

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
    filelist = glob.glob(exp_s)
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
    filelist = glob.glob(exp_s)
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
        rtol=1e-7,
        atol=1e-6,
        err_msg="values in plot do not match saved values",
    )


def test_ts_multi():
    """Test plotting time series of min and max."""
    domain = "ch"
    filelist = glob.glob(exp_s)
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
    np.testing.assert_allclose(
        da_mean.values,
        test_mean,
        rtol=1e-7,
        atol=1e-6,
        err_msg="computed values do not match saved values for mean",
    )
    np.testing.assert_allclose(
        da_max.values,
        test_max,
        rtol=1e-7,
        atol=1e-6,
        err_msg="computed values do not match saved values for max",
    )


def test_hist():
    """Test plotting one histogram."""
    domain = "ch"
    filelist = glob.glob(exp_s)
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
    np.testing.assert_allclose(
        counts,
        test_bar,
        rtol=0,
        atol=0,
        err_msg="computes bar values do not match saved values",
    )


def test_hist_multiexps():
    """Test plotting two histograms, and deaggregate."""
    min_bin = 0
    max_bin = 50
    nbins = 50
    da_dict = {}
    for e, n in zip([exp_1, exp_2], ["ICON1", "ICON2"]):
        filelist = glob.glob(e)
        da = prepare_masked_da(
            filelist,
            "TOT_PREC",
            deagg=True,
        )
        da_dict[n] = da
    plot_histograms(da_dict, min_bin=min_bin, max_bin=max_bin, nbins=nbins, save=False)
    np.testing.assert_allclose(
        plt.gca().containers[0].datavalues,
        test_bar_1,
        rtol=0,
        atol=0,
        err_msg="computes bar values do not match saved values for exp1",
    )
    np.testing.assert_allclose(
        plt.gca().containers[1].datavalues,
        test_bar_2,
        rtol=0,
        atol=0,
        err_msg="computes bar values do not match saved values for exp2",
    )
