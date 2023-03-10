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
from icon_timeseries.prepare_data import prepare_nn

# single experiment, deterministic
test_single = "/store/s83/cmerker/test_data/icon_timeseries/data/test_plots"
gdf = os.path.join(test_single, "ICON-1E_DOM01.nc")
exp_s = os.path.join(test_single, "lfff000[0-3]0000")

# two experiments, deterministic
test_exps = "/store/s83/cmerker/test_data/icon_timeseries/data/test_ens"
gdf1 = os.path.join(test_exps, "ICON-1E_DOM01.nc")
gdf2 = os.path.join(test_exps, "ICON-2E_DOM01.nc")
exp_1 = os.path.join(test_exps, "ICON-CH1-EPS", "004", "lfff000[0-4]0000")
exp_2 = os.path.join(test_exps, "ICON-CH2-EPS", "004", "lfff000[0-4]0000")

# two experiments, ensemble, day time for
exp_1_ensday = os.path.join(test_exps, "ICON-CH1-EPS", "*", "lfff001[1-6]0000")
exp_2_ensday = os.path.join(test_exps, "ICON-CH2-EPS", "*", "lfff001[1-6]0000")

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
test_nn = np.array(
    [
        [113.45312, 473.73438, 450.90625, 415.73438, 326.5625, 191.45312],
        [111.28516, 448.6914, 478.0625, 428.5, 322.875, 188.10938],
        [112.00391, 468.16016, 440.6328, 414.4922, 323.5, 191.29688],
        [112.85938, 485.9375, 456.40625, 416.92188, 324.04688, 193.73438],
        [112.13672, 443.26172, 457.67188, 422.64062, 324.79688, 187.25],
        [110.22266, 431.08203, 462.26562, 415.1875, 314.71875, 183.29688],
        [113.21875, 474.48438, 450.29688, 409.96875, 320.9375, 191.03125],
        [112.39062, 430.95312, 481.10156, 429.5078, 327.92188, 188.96875],
        [112.72266, 497.5664, 451.35938, 415.5625, 323.14062, 192.54688],
        [111.82031, 475.5703, 459.0078, 424.41406, 319.6953, 189.41406],
        [113.21484, 487.23047, 466.03125, 423.53125, 326.8672, 192.47656],
        [112.01562, 479.1875, 455.8828, 415.4297, 326.5078, 193.57031],
        [110.32422, 434.13672, 477.08594, 428.5703, 326.875, 193.14062],
        [110.32031, 435.0703, 455.6797, 419.22656, 320.64062, 177.60938],
        [110.80078, 452.09766, 462.92188, 416.57812, 325.4922, 190.72656],
        [113.08203, 490.28516, 428.28125, 399.42188, 314.2422, 189.99219],
        [112.0, 495.20312, 466.34375, 414.71875, 323.0625, 194.17188],
        [111.98047, 457.3086, 443.0078, 411.52344, 325.59375, 186.53125],
        [110.33984, 465.46484, 474.6328, 429.3828, 326.1953, 192.46094],
        [107.17188, 429.0625, 474.89062, 426.875, 322.58594, 188.94531],
        [110.64844, 501.8203, 426.65625, 407.09375, 303.85938, 184.4375],
        [109.75, 502.0, 420.0078, 354.4297, 307.1328, 183.55469],
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
    """Test plotting one time series of domain mean/max."""
    domain = "ch"
    filelist = glob.glob(exp_s)
    da_mean, _ = prepare_meanmax(
        filelist,
        "T",
        80,
        gdf,
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
    """Test plotting one time series of domain mean/max."""
    domain = "ch"
    filelist = glob.glob(exp_s)
    da_mean, _ = prepare_meanmax(
        filelist,
        "T",
        80,
        gdf,
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
    """Test plotting time series of domain mean and max."""
    domain = "ch"
    filelist = glob.glob(exp_s)
    da_mean, da_max = prepare_meanmax(
        filelist,
        "T",
        80,
        gdf,
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
        gdf,
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


def test_ts_nn_multi():
    """Test plotting time series of single point values, two experiments ensemble."""
    # get data from test files
    da_dict = {"values": {}}
    for e, n, gd in zip([exp_1_ensday, exp_2_ensday], ["ICON1", "ICON2"], [gdf1, gdf2]):
        filelist = glob.glob(e)
        da = prepare_nn(
            filelist,
            "ASOB_S",
            "8.54,47.40",
            gridfile=gd,
            deagg=True,
        )
        da_dict["values"][n] = da
    # plot data from test files
    plot_ts_multiple(da_dict, save=False)
    # get values from plot lines
    plot_vals = np.array(
        [li.get_xydata()[:, 1] for li in plt.gcf().get_axes()[0].lines[:]]
    )
    # get values from data
    comp_vals = np.hstack(
        [da_dict["values"]["ICON1"].values, da_dict["values"]["ICON2"].values]
    ).T
    # test data from plot vs. input data
    np.testing.assert_array_equal(
        plot_vals,
        comp_vals,
        "y values in plot do not match input values",
    )
    # test input data vs. saved reference data
    np.testing.assert_allclose(
        comp_vals,
        test_nn,
        rtol=1e-7,
        atol=1e-6,
        err_msg="computed values do not match saved values",
    )
