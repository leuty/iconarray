"""Temporally deaggregate values in xarray.Datasets."""
# Standard library
import logging

# Third-party
import numpy as np
import xarray as xr


def deaggregate(da: xr.DataArray) -> xr.DataArray:
    """Deaggregate accumulated output values over valid times.

    Apply deagg_sum or deaverage over groups with identical initial times.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-aggregated values. Needs valid_time as dimension
        and valid_time and ini_time as coordinate.

    """
    logging.info("Data deaggregation: Searching for accumulation info.")

    # check if time dimension is at least 2 long
    _check_time_dim(da)

    if da.attrs["GRIB_stepType"] == "accum":
        da = da.groupby("ini_time").map(deagg_sum)
    elif da.attrs["GRIB_stepType"] == "avg":
        da = da.groupby("ini_time").map(deaverage)
    else:
        logging.error(
            "No deaggregation method is implemented for %s", da.attrs["GRIB_stepType"]
        )
    return da


def deaverage(da: xr.DataArray) -> xr.DataArray:
    """Deaverage (over valid_time).

    x_{n} = n * x_{n} + (n - 1) * x_{n-1}

    Lead time +0h and +1h are ignored, because there is nothing to deaggreagate.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-averaged values. Needs valid_time as dimension and
        valid_time as a coordinate.

    """
    logging.info("Data deaggregation: de-averaging")

    # define time step and check for irregularities
    dt = _check_time_steps(da)

    # get time step number
    n_fcst = ((da.valid_time - da.ini_time[0]) / dt).astype(np.int32)
    # ignore lead time 0
    time = da.valid_time[n_fcst >= 1]
    n_fcst = n_fcst[n_fcst >= 1]

    # deaverage the data leaving out the first considered lead time
    deavg = da.copy()
    da_shift = da.sel(valid_time=time[:-1])
    da_shift = da_shift.assign_coords({"valid_time": time[1:]})
    deavg.loc[{"valid_time": time[1:]}] = da.sel(valid_time=time[1:]) * n_fcst[
        1:
    ] - da_shift.sel(valid_time=time[1:]) * (n_fcst[1:] - 1)

    deavg.attrs["GRIB_stepType"] = "instant"
    return deavg


def deagg_sum(da: xr.DataArray) -> xr.DataArray:
    """Deaggregate a sum (over valid_time).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-aggregated (summed) values. Needs valid_time as dimension
        and valid_time as a coordinate.

    """
    logging.info("Data deaggregation: de-accumulation")

    # check for irregularities in time steps
    dt = _check_time_steps(da)

    # ignore lead time 0
    time = da.valid_time[da.valid_time >= (da.ini_time[0] + dt)]

    # deaggregation from LT 1h, we can include LT 0 in the subtrahend, should be ~0
    deaggd = da.copy()
    deaggd.loc[{"valid_time": time[1:]}] = da.sel(valid_time=time).diff(
        dim="valid_time", n=1, label="upper"
    )
    deaggd.attrs["GRIB_stepType"] = "instant"
    return deaggd


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
        minimal occurring time difference between forecast time in data set

    """
    # get steps between available forecast times
    dt = np.unique(da.valid_time[1:].values - da.valid_time[:-1].values)
    # check for irregularities
    if len(dt) != 1:
        logging.warning(
            "Found irregular time spacing in data. Deaggregation might be wrong."
        )
    dt = min(dt)
    logging.info("Detected time step: %d", dt)

    return dt
