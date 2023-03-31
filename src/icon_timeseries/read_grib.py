"""Read GRIB files."""
# Standard library
import logging
import sys
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Mapping

# Third-party
import numpy as np
import xarray as xr

# Local
from .deaggregate import deaggregate

# keep attributes on xarray Datasets and DataArrays
xr.set_options(keep_attrs=True)


# pylint: disable=too-many-arguments
def get_var(
    filelist: List[str],
    varname: str,
    level: float | None = None,
    deagg: bool = False,
    chunks: Dict[str, int] | None = None,
) -> xr.DataArray:
    """Get a DataArray of a model quantity, deaggregate if required.

    Parameters
    ----------
    filelist : list(str)
        list of files to read
    varname : str
        GRIB shortName of variable to extract
    level : int, optional
        model level value
    deagg : bool, optional
        Deaggregation of variable, de-averaging and de-accumulation are currently
        available
    chunks : Dict(str, int), optional
        chunk size for each dimension to be loaded.

    Returns
    -------
    da : xarray.DataArray
        DataArray with model data

    """
    # read and time
    tstart = time.perf_counter()
    da = var_from_files(
        filelist,
        varname,
        level,
        parallel=True,
        chunks=chunks,
    )
    tend = time.perf_counter()
    telapsed = tend - tstart
    logging.info("reading time elapsed: %f", telapsed)

    # data deaggregation
    if not deagg:
        pass
    else:
        da = deaggregate(da)

    return da


# pylint: enable=too-many-arguments


# pylint: disable=too-many-arguments
def var_from_files(
    filelist: List[str],
    varname: str,
    level: float | None = None,
    parallel: bool = False,
    chunks: Dict[str, int] | None = None,
) -> xr.DataArray:
    """Read a variable from GRIB file(s) into an xarray.DataArray.

    Parameters
    ----------
    filelist : list(str)
        list of files to read
    varname : str
        GRIB shortName of variable to extract
    level : float, optional
        model level value, no selection if None
    parallel : bool, optional
        parallelise the reading with dask.
    chunks : Dict(str, int), optional
        chunk size for each dimension to be loaded.

    Returns
    -------
    da : xarray.DataArray
        output data

    """
    filelist.sort()
    logging.info("reading %d files", len(filelist))
    logging.info("files: %s", filelist)

    # define arguments for open_mfdataset and the cfgrib engine
    backend_kwargs = {
        "indexpath": "",
        "errors": "ignore",
        "filter_by_keys": {"shortName": varname},
    }
    kwargs: Mapping[
        str, Any
    ] = {  # type needed to avoid mypy arg-type error in open_mfdataset
        "engine": "cfgrib",
        "chunks": chunks,
        "backend_kwargs": backend_kwargs,
        "encode_cf": ("time", "geography", "vertical"),
    }

    # open the datasets
    ds = _open_icondataset(
        filelist,
        parallel=parallel,
        **kwargs,
    )

    # selection
    try:
        da = ds[varname]
    except KeyError:
        logging.error(
            "Cannot find %s in data. Make sure your ecCodes environment "
            "is loaded correctly, e.g. did you run ./setup_grib_env.sh?",
            varname,
        )
        sys.exit()
    if level is not None:
        try:
            da = da.sel({da.GRIB_typeOfLevel: level}, method="nearest", tolerance=1e-09)
            da.attrs["level"] = level
        except KeyError as e:
            logging.error(
                "level %s not found in data, available levels: %s",
                level,
                ", ".join([f"{x:.2f}" for x in da[da.GRIB_typeOfLevel].values]),
            )
            raise KeyError(e) from e

    try:
        if da.number.shape[0] > 1:
            # this is a bit hacky: we first copy the coordinate (not dim) time
            # to ini_time and then manipulate time, which is dropped during the
            # unstack. The rename at the bottom is necessary to catch all non-
            # ensemble cases.
            logging.info("Found multiple ensemble members. Reshaping ensemble data.")
            da = da.assign_coords({"ini_time": da.time})
            # type ignore on the following line: known for some versions of xarray,
            # see e.g. https://github.com/pydata/xarray/issues/6576
            da = da.set_index(
                {"time": ["valid_time", "number"]}, append=False  # type:ignore
            )
            # check if valid_time is unique
            _check_index_isunique(da)
            da = da.unstack("time")
            da = _reshape_initime(da)
        else:
            logging.info("Only one ensemble member found. Continuing.")
    except IndexError:
        logging.info("Dimension for ensemble size empty. Continuing.")
    except AttributeError:
        logging.info("No dimension for ensemble information found. Continuing.")

    if "time" in da.coords:  # rename time for all non-ensemble cases.
        da = da.assign_coords({"ini_time": ("time", da.coords["time"].copy().values)})
        da = _fix_time_name(da)

    return da


# pylint: enable=too-many-arguments


def _open_icondataset(
    filelist: List[str],
    concat_dim: str = "time",
    combine: Literal["by_coords", "nested"] = "nested",
    parallel: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """Open a dataset from a list of ICON / COSMO GRIB files.

    Wrapper for xr.open_mfdataset for better exception handling.

    Parameter
    ---------
    filelist : list
        List of files to open
    concat_dim : str
        Dimension to concatenate along
    combine : str
        How to combine the files
    parallel : bool
        Whether to parallelize the reading
    kwargs : dict
        Additional keyword arguments passed to xr.open_mfdataset

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the data from the files

    Raises
    ------
    ValueError
        If the ensemble dimension is not present in all files

    """
    try:
        ds = xr.open_mfdataset(
            filelist,
            concat_dim=concat_dim,
            combine=combine,
            parallel=parallel,
            **kwargs,
        )
    except ValueError as e:
        if str(e).startswith("'number' not present in all datasets"):
            logging.error(
                "The ensemble dimension is not present in all files. "
                "Check your file list or the GRIB encoding. Or do you maybe "
                "read in the constant file? "
            )
        raise ValueError(e) from e
    return ds


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
    if "valid_time" not in da.dims:
        try:
            da = da.swap_dims({"time": "valid_time"})
        except ValueError:
            logging.warning("Time dimension is empty. Only ok if this is a grid file.")

    return da


def _reshape_initime(da: xr.DataArray) -> xr.DataArray:
    """Reshape initial time to 1D array with length valid_time."""
    index_ensdim = da.ini_time.dims.index("number")
    initime_reduced = np.unique(da.ini_time, axis=index_ensdim)
    initime_reduced = np.squeeze(initime_reduced)
    del da.coords["ini_time"]
    da = da.assign_coords({"ini_time": ("valid_time", initime_reduced)})
    if len(da.ini_time.shape) > 1:
        raise RuntimeError("Ensemble members have non-identical initial times.")

    return da


def _check_index_isunique(da: xr.DataArray) -> None:
    """Check if time steps in the data sets are unique.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray must have valid_time as an attribute (variable, coord, dim)

    Raises
    ------
    NotImplementedError
        if time steps are not unique

    """
    # if da.index.shape != da.index.unique().shape:
    if da.indexes["time"].shape != da.indexes["time"].unique().shape:
        logging.error("Either valid time or ens members in data are not unique.")
        raise NotImplementedError(
            "Non-unique combinations of members and valid times are not supported. "
            "Maybe you have overlapping forecasts?"
        )
