"""Data pre-processing for visualisation."""
# Standard library
import logging
import time
from typing import Dict
from typing import List
from typing import Tuple

# Third-party
import xarray as xr

# Local
from .deaggregate import deagg_sum
from .deaggregate import deaverage
from .handle_grid import get_grid
from .read_grib import var_from_files
from .utils import check_grid
from .utils import ind_from_nn
from .utils import mask_domain
from .utils import nearest_xy
from .utils import parse_coords


# pylint: disable=too-many-arguments
def prepare_meanmax(
    filelist: List[str],
    varname: str,
    level: int | None = None,
    gridfile: str | None = None,
    domain: str = "all",
    deagg: bool = False,
    chunks: Dict[str, int] | None = None,
    dask_nworkers: int | None = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Get the domain average and domain maximum of a model quantity.

    Parameters
    ----------
    filelist : list(str)
        list of files to read
    varname : str
        GRIB shortName of variable to extract
    level : int
        model level index
    gridfile : str, optional
        ICON grid file, needed for unstructured grid
    domain : str
        domain to consider, please define in domains.yaml. default is whole domain
    deagg : bool
        Deaggregation of variable, de-averaging and de-accumulation are currently
        available
    chunks : Dict(str, int), optional
        chunk size for each dimension to be loaded.
    dask_nworkers : int, optional
        if set, data reading is done in parallel using dask_nworkers workers

    Returns
    -------
    da_mean : xarray.DataArray
        domain average
    da_max : xarray.DataArray
        domain maximum

    """
    # get domain
    if gridfile:
        gd = get_grid(gridfile)
    # check compatibility of grid, domain and data
    if domain != "all" and gridfile:
        check_grid(
            filelist, gd, varname, level, chunks=chunks, dask_nworkers=dask_nworkers
        )

    # read the data
    tstart = time.perf_counter()
    da = var_from_files(
        filelist,
        varname,
        level,
        parallel=True,
        chunks=chunks,
        dask_nworkers=dask_nworkers,
    )
    tend = time.perf_counter()
    telapsed = tend - tstart
    logging.info("reading time elapsed: %f", telapsed)

    # data deaggregation
    if not deagg:
        pass
    elif deagg and da.attrs["GRIB_stepType"] == "accum":
        da = deagg_sum(da)
    elif deagg and da.attrs["GRIB_stepType"] == "avg":
        da = deaverage(da)
    else:
        logging.error(
            "No deaggregation method is implemented for %s",
            da.attrs["GRIB_stepType"]
        )

    # apply domain mask if domain is set
    if domain != "all" and "gd" in locals():
        da = mask_domain(da, domain, gd)
    elif domain != "all" and "gd" not in locals():
        da = mask_domain(da, domain)

    # compute average and maximum
    if da.attrs["GRIB_gridType"] == "rotated_ll":
        da_mean = da.mean(dim=["x", "y"], skipna=True).compute()
        da_max = da.max(dim=["x", "y"], skipna=True).compute()
    else:
        da_mean = da.mean(dim="values", skipna=True).compute()
        da_max = da.max(dim="values", skipna=True).compute()

    return da_mean, da_max


# pylint: disable=too-many-locals
def prepare_nn(
    filelist: List[str],
    varname: str,
    lonlat: str,
    level: int | None = None,
    gridfile: str | None = None,
    deagg: bool = False,
    chunks: Dict[str, int] | None = None,
    dask_nworkers: int | None = None,
) -> xr.DataArray:
    """Get the domain average and domain maximum of a model quantity.

    Parameters
    ----------
    filelist : list(str)
        list of files to read
    varname : str
        GRIB shortName of variable to extract
    lonlat : str
        Coordinates to consider for nn lookup. Format ('lon,lat').
    level : int
        model level index
    gridfile : str, optional
        ICON grid file, needed for unstructured grid
    deagg : bool
        Deaggregation of variable, de-averaging and de-accumulation are currently
        available
    chunks : Dict(str, int), optional
        chunk size for each dimension to be loaded.
    dask_nworkers : int, optional
        if set, data reading is done in parallel using dask_nworkers workers

    Returns
    -------
    da_mean : xarray.DataArray
        Nearest Neighbour Values

    """
    # read the data
    tstart = time.perf_counter()
    da = var_from_files(
        filelist,
        varname,
        level,
        parallel=True,
        chunks=chunks,
        dask_nworkers=dask_nworkers,
    )
    tend = time.perf_counter()
    telapsed = tend - tstart
    logging.info("reading time elapsed: %f", telapsed)

    # get grid
    if gridfile:
        gd = get_grid(gridfile)
        # check compatibility of grid and data
        check_grid(
            filelist, gd, varname, level, chunks=chunks, dask_nworkers=dask_nworkers
        )

    # data deaggregation
    if not deagg:
        pass
    elif deagg and da.attrs["GRIB_stepType"] == "accum":
        da = deagg_sum(da)
    elif deagg and da.attrs["GRIB_stepType"] == "avg":
        da = deaverage(da)
    else:
        logging.error(
            "No deaggregation method is implemented for %s",
            da.attrs["GRIB_stepType"]
        )

    lon, lat = parse_coords(lonlat)
    if "gd" in locals():  # unstructured grid
        index = ind_from_nn(gd.cx, gd.cy, lon, lat)
        da_nn = da.isel({"values": index})
    else:  # rotated pole
        x, y = nearest_xy(da.longitude.values, da.latitude.values, lon, lat)
        da_nn = da.isel(y=y, x=x)
    return da_nn


# pylint: enable=too-many-arguments,too-many-locals
