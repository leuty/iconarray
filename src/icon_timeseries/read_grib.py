"""Read GRIB files."""
# Standard library
import logging
import sys
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping

# Third-party
import numpy as np
import xarray as xr
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask.distributed import performance_report

# keep attributes on xarray Datasets and DataArrays
xr.set_options(keep_attrs=True)


# pylint: disable=too-many-arguments
def var_from_files(
    filelist: List[str],
    varname: str,
    level: int | None = None,
    parallel: bool = False,
    chunks: Dict[str, int] | None = None,
    dask_nworkers: int | None = None,
) -> xr.DataArray:
    """Read a variable from GRIB file(s) into an xarray.DataArray.

    Parameters
    ----------
    filelist : list(str)
        list of files to read
    varname : str
        GRIB shortName of variable to extract
    level : int, optional
        model level index, no selection if None
    parallel : bool, optional
        parallelise the reading with dask.
    chunks : Dict(str, int), optional
        chunk size for each dimension to be loaded.
    dask_nworkers : int, optional
        if set, data reading is done in parallel using dask_nworkers workers

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

    # setup the dask cluster if requested
    if dask_nworkers:
        cluster = LocalCluster()
        cluster.scale(dask_nworkers)
        client = Client(cluster)

        u_id = time.time()
        dask_report = f"dask-report-{u_id}.html"
        logging.info(
            "dask report is being prepared: %s, %s", dask_report, client.dashboard_link
        )

        with performance_report(filename=dask_report):
            ds = xr.open_mfdataset(
                filelist,
                concat_dim="time",
                combine="nested",
                parallel=parallel,
                **kwargs,
            )
    else:
        ds = xr.open_mfdataset(
            filelist, concat_dim="time", combine="nested", parallel=parallel, **kwargs
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
    if level:
        try:
            da = da.loc[{da.GRIB_typeOfLevel: level}]
            da.attrs["level"] = level
        except KeyError:
            logging.error("level not found in data")
            sys.exit()

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
            da = da.unstack("time")
            da = _reshape_initime(da)
        else:
            logging.info("Only one ensemble member found. Continuing.")
    except IndexError:
        logging.info("Dimension for ensemble size empty. Continuing.")
    except AttributeError:
        logging.info("No dimension for ensemble information found. Continuing.")

    if "time" in da.coords:  # rename time for all non-ensemble cases.
        da = da.assign_coords({"ini_time": da.time})
        da = da.set_index({"time": "ini_time"})
        da = da.reset_index("time")
        da = da.reset_coords("time", drop=True)

    return da


# pylint: enable=too-many-arguments


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
