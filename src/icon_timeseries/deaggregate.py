"""Temporally deaggregate values in xarray.Datasets."""
# Standard library
import logging
import sys

# Third-party
import numpy as np
import xarray as xr


def deaverage(da: xr.DataArray) -> xr.DataArray:
    """Deaverage (over valid_time).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-averaged values. Needs valid_time as dimension.

    """
    try:
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    except KeyError:
        da = da.swap_dims({"time": "valid_time"})
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    subtrahend = subtrahend.assign_coords({"valid_time": da.valid_time[2:]})
    dt = da.valid_time[1] - da.valid_time[0]  # improve with unique and catch irregular
    n_fcst = ((da.valid_time[2:] - da.valid_time[0]) / dt).astype(np.int32)  # ns to h
    deavd = da
    deavd.loc[{"valid_time": da.valid_time[2:]}] = da * n_fcst - subtrahend * (
        n_fcst - 1
    )
    deavd.attrs["GRIB_stepType"] = "instant"
    return deavd


def deagg_sum(da: xr.DataArray) -> xr.DataArray:
    """Deaggregate a sum (over valid_time).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-aggregated (summed) values. Needs valid_time as dimension.

    """
    try:
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    except KeyError:
        da = da.swap_dims({"time": "valid_time"})
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    subtrahend = subtrahend.assign_coords({"valid_time": da.valid_time[2:]})
    deaggd = da
    deaggd.loc[{"valid_time": da.valid_time[2:]}] = (
        da.loc[{"valid_time": da.valid_time[2:]}] - subtrahend
    )
    deaggd.attrs["GRIB_stepType"] = "instant"
    return deaggd


def _check_time_dim(da: xr.DataArray) -> None:
    """Check if time dimension is longer than one.

    Raise Error if shape of da.valid_time is empty tuple or 1.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray must have valid_time as an attribute (variable, coord, dim)

    """
    if da.valid_time.shape == ():
        logging.error(
            "The time dimension of the given data must be longer"
            "than one for deaggregation."
        )
        sys.exit()
    elif da.valid_time.shape[0] < 2:
        logging.error(
            "The time dimension of the given data must be longer "
            "than one for deaggregation."
        )
        sys.exit()
    else:
        pass
