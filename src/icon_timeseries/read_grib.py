"""Read GRIB files."""
# Standard library
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping

# Third-party
import xarray as xr
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask.distributed import performance_report


def var_from_files(
    filelist: List[str],
    varname: str,
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
    parallel : bool, optional
        parallelise the reading with dask.
    chunks : Dict(str, int)
        chunk size for each dimension to be loaded.
    dask_nworkers : int, optional
        if set, data reading is done in parallel using dask_nworkers workers

    Returns
    -------
    da : xarray.DataArray
        output data

    """
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

    da = ds[varname]
    return da
