"""Tests for plots."""
# Standard library
import glob
import os
import pickle

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

# First-party
from icon_timeseries.handle_grid import get_grid
from icon_timeseries.plot import plot_histograms
from icon_timeseries.plot import plot_on_map
from icon_timeseries.plot import plot_ts
from icon_timeseries.plot import plot_ts_multiple
from icon_timeseries.prepare_data import prepare_masked_da
from icon_timeseries.prepare_data import prepare_meanmax
from icon_timeseries.prepare_data import prepare_nn
from icon_timeseries.prepare_data import prepare_time_avg

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
# update the reference with:
#   import pickle
#   f = open(pickle_file_name, "wb")
#   pickle.dump(data, f)
#   f.close()
with open(os.path.join(test_single, "test_mean.pckl"), "rb") as f:
    test_mean = pickle.load(f)
with open(os.path.join(test_single, "test_max.pckl"), "rb") as f:
    test_max = pickle.load(f)
with open(os.path.join(test_single, "test_bar.pckl"), "rb") as f:
    test_bar = pickle.load(f)
with open(os.path.join(test_exps, "test_bar_1.pckl"), "rb") as f:
    test_bar_1 = pickle.load(f)
with open(os.path.join(test_exps, "test_bar_2.pckl"), "rb") as f:
    test_bar_2 = pickle.load(f)
with open(os.path.join(test_exps, "test_nn.pckl"), "rb") as f:
    test_nn = pickle.load(f)
with open(os.path.join(test_single, "test_time_avg.pckl"), "rb") as f:
    test_time_avg = pickle.load(f)

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
        err_msg="computed bar values do not match saved values",
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
        err_msg="computed bar values do not match saved values for exp1",
    )
    np.testing.assert_allclose(
        plt.gca().containers[1].datavalues,
        test_bar_2,
        rtol=0,
        atol=0,
        err_msg="computed bar values do not match saved values for exp2",
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


def test_time_avg_plot():
    """Test plotting a temporal average on a map."""
    filelist = glob.glob(exp_s)
    ds = prepare_time_avg(
        filelist,
        "T",
        1,
    )
    gd = get_grid(gdf)
    _, _ = plot_on_map(
        ds["T"],
        gd,
        save=False,
    )
    np.testing.assert_allclose(
        plt.gca().collections[2].get_paths()[0].vertices,
        test_time_avg.vertices,
        rtol=1e-7,
        atol=1e-6,
        err_msg="computed Path vertices values do not match saved values",
    )
    np.testing.assert_allclose(
        plt.gca().collections[2].get_paths()[0].codes,
        test_time_avg.codes,
        rtol=1e-7,
        atol=1e-6,
        err_msg="computed Path codes values do not match saved values",
    )
