"""Tests for deaggregation."""
# Standard library
import datetime

# Third-party
import numpy as np
import xarray as xr

# First-party
from icon_timeseries.deaggregate import deagg_sum
from icon_timeseries.deaggregate import deaggregate
from icon_timeseries.deaggregate import deaverage


def _create_test_da() -> xr.DataArray:
    """Create simple sample DataArray"""
    smpl_data = np.random.rand(5, 5)
    t = np.arange("2022-11-28T00:00", "2022-11-28T05:00", dtype="datetime64[h]")
    t_start = np.empty(t.shape, dtype="datetime64[h]")
    t_start[:] = t[0]
    x = np.arange(0, 5, dtype="int32")
    smpl_coords = (t, x)
    smpl_da = xr.DataArray(smpl_data, dims=["valid_time", "x"], coords=smpl_coords)
    smpl_da = smpl_da.assign_coords(ini_time=("valid_time", t_start))
    return smpl_da


def test_deaggregate():
    """Test deaggregation."""
    test_da = _create_test_da()
    ini_t1 = np.repeat(
        np.array([datetime.datetime(2022, 11, 28, hour=0)], dtype="datetime64[h]"), 3
    )
    ini_t2 = np.repeat(
        np.array([datetime.datetime(2022, 11, 28, hour=3)], dtype="datetime64[h]"), 2
    )
    ini_times = np.concatenate((ini_t1, ini_t2))
    test_da = test_da.drop_vars("ini_time")
    test_da = test_da.assign_coords(
        {"ini_time": ("valid_time", ini_times)}  # type:ignore
    )
    agg_da = test_da.copy()
    agg_da[2] = agg_da[1, :] + test_da[2, :]
    agg_da = agg_da.assign_attrs({"GRIB_stepType": "accum"})
    deagg_da = deaggregate(agg_da)
    np.testing.assert_array_almost_equal(deagg_da, test_da, decimal=14)


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