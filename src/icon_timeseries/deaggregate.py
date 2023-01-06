"""Temporally deaggregate values in xarray.Datasets."""
# Standard library
import logging

# Third-party
import numpy as np
import xarray as xr


def deaverage(da: xr.DataArray) -> xr.DataArray:
    """Deaverage (over valid_time).

    x_{n} = n * x_{n} + (n - 1) * x_{n-1}

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-averaged values. Needs valid_time as dimension.

    """
    logging.info("Data deaggregation: de-averaging")
    # fix name of time dimension
    da = _fix_time_name(da)
    # check if time dimension is at least 2 long
    _check_time_dim(da)

    # define time step and check for irregularities
    dt = _check_time_steps(da)

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
    logging.info("Data deaggregation: de-accumulation")
    # fix name of time dimension
    da = _fix_time_name(da)
    # check if time dimension is at least 2 long
    _check_time_dim(da)

    # check for irregularities in time steps
    _ = _check_time_steps(da)

    subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    subtrahend = subtrahend.assign_coords({"valid_time": da.valid_time[2:]})
    deaggd = da
    deaggd.loc[{"valid_time": da.valid_time[2:]}] = (
        da.loc[{"valid_time": da.valid_time[2:]}] - subtrahend
    )
    deaggd.attrs["GRIB_stepType"] = "instant"
    return deaggd


def _fix_time_name(da: xr.DataArray) -> xr.DataArray:
    """Check if valid_time is present, otherwise rename time dimension.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to check

    Returns
    -------
    da : xarray.DataArray
        DataArray to check

    """
    if not "valid_time" in da.dims:
        da = da.swap_dims({"time": "valid_time"})

    return da


def _check_time_dim(da: xr.DataArray) -> None:
    """Check if time dimension is longer than one.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray must have valid_time as an attribute (variable, coord, dim)

    Raises
    ------
    IndexError
        if shape of da.valid_time is empty tuple or 1

    """
    if da.valid_time.shape == ():
        logging.error(
            "The time dimension of the given data must be longer"
            "than one for deaggregation."
        )
        raise IndexError
    elif da.valid_time.shape[0] < 2:
        logging.error(
            "The time dimension of the given data must be longer "
            "than one for deaggregation."
        )
        raise IndexError
    else:
        pass


def _check_time_steps(da: xr.DataArray) -> float:
    """Check if time steps in the data sets are irregular.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray must have valid_time as an attribute (variable, coord, dim)

    Returns
    -------
    dt : float
        minimal occuring time difference between forecast time in data set

    """
    # get steps between available forecast times
    dt = np.unique(da.valid_time[1:].values - da.valid_time[:-1].values)
    # check for irregularities
    if len(dt) != 1:
        logging.warning("Found irregular time spacing in data. Deaggregation might "
        "be wrong.")
    dt = min(dt)
    logging.info("Detected time step: %d", dt)

    return dt
