"""Data pre-processing for visualisation."""
# Standard library
from typing import Dict
from typing import List
from typing import Tuple

# Third-party
import xarray as xr

# Local
from .handle_grid import get_grid
from .read_grib import get_var
from .utils import check_grid
from .utils import ind_from_nn
from .utils import mask_domain
from .utils import nearest_xy
from .utils import parse_coords


# pylint: disable=too-many-arguments
def prepare_meanmax(
    filelist: List[str],
    varname: str,
    level: float | None = None,
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
    level : float
        model level value
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
    da = prepare_masked_da(
        filelist,
        varname,
        level,
        gridfile,
        domain,
        deagg,
        chunks,
        dask_nworkers,
    )

    # compute average and maximum
    if da.attrs["GRIB_gridType"] == "rotated_ll":
        da_mean = da.mean(dim=["x", "y"], skipna=True).compute()
        da_max = da.max(dim=["x", "y"], skipna=True).compute()
    else:
        da_mean = da.mean(dim="values", skipna=True).compute()
        da_max = da.max(dim="values", skipna=True).compute()

    return da_mean, da_max


# pylint: enable=too-many-arguments


# pylint: disable=too-many-arguments, too-many-locals
def prepare_nn(
    filelist: List[str],
    varname: str,
    lonlat: str,
    level: float | None = None,
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
    level : float
        model level value
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
    # get grid
    if gridfile:
        gd = get_grid(gridfile)
        # check compatibility of grid and data
        check_grid(
            filelist, gd, varname, level, chunks=chunks, dask_nworkers=dask_nworkers
        )

    # read the data
    da = get_var(
        filelist,
        varname,
        level,
        deagg=deagg,
        chunks=chunks,
        dask_nworkers=dask_nworkers,
    )

    lon, lat = parse_coords(lonlat)
    if "gd" in locals():  # unstructured grid
        index = ind_from_nn(gd.cx, gd.cy, lon, lat)
        da_nn = da.isel({"values": index})
    else:  # rotated pole
        x, y = nearest_xy(da.longitude.values, da.latitude.values, lon, lat)
        da_nn = da.isel(y=y, x=x)
    return da_nn


# pylint: enable=too-many-arguments, too-many-locals


# pylint: disable=too-many-arguments
def prepare_masked_da(
    filelist: List[str],
    varname: str,
    level: float | None = None,
    gridfile: str | None = None,
    domain: str = "all",
    deagg: bool = False,
    chunks: Dict[str, int] | None = None,
    dask_nworkers: int | None = None,
) -> xr.DataArray:
    """Get a (domain) masked DataArray of a model quantity.

    Parameters
    ----------
    filelist : list(str)
        list of files to read
    varname : str
        GRIB shortName of variable to extract
    level : int, optional
        model level value
    gridfile : str, optional
        ICON grid file, needed for unstructured grid
    domain : str, optional
        domain to consider for masking
    deagg : bool, optional
        Deaggregation of variable, de-averaging and de-accumulation are currently
        available
    chunks : Dict(str, int), optional
        chunk size for each dimension to be loaded.
    dask_nworkers : int, optional
        if set, data reading is done in parallel using dask_nworkers workers

    Returns
    -------
    da : xarray.DataArray
        masked DataArray

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
    da = get_var(
        filelist,
        varname,
        level,
        deagg=deagg,
        chunks=chunks,
        dask_nworkers=dask_nworkers,
    )

    # apply domain mask if domain is set
    if domain != "all" and "gd" in locals():
        da = mask_domain(da, domain, gd)
    elif domain != "all" and "gd" not in locals():
        da = mask_domain(da, domain)

    return da

    # pylint: enable=too-many-arguments
