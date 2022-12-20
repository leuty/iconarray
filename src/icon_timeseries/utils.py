"""Utils for data pre-processing."""
# Standard library
import logging
import sys
from typing import Dict
from typing import List
from typing import Tuple

# Third-party
import numpy as np
import xarray as xr

# Local
from .handle_grid import get_domain
from .handle_grid import IconGrid
from .handle_grid import points_in_domain
from .read_grib import var_from_files


def nearest_xy(lon2d: np.ndarray, lat2d: np.ndarray, lon: float, lat: float):
    """Find the nearest x, y values for given lat, lon.

    Parameter
    ---------
    lon2d : np.ndarray (2d)
    lat2d : np.ndarray (2d)
    lon : float
    lat : float

    """
    dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
    yx = np.where(dist == np.min(dist))
    y = yx[0][0]
    x = yx[1][0]
    return x, y


def ind_from_nn(lats, lons, lat, lon):
    """Find the nearest neighbouring index to given location.

    Copy-pasted from plot_profile/plot_icon/get_icon.py (sawrry ...)
    Args:
        lats (2d array):            Latitude grid
        lons (2d array):            Longitude grid
        lat (float):                Latitude of location
        lon (float):                Longitude of location

    Returns:
        int     Index of nearest grid point.

    """
    dist = [
        np.sqrt((lats[i] - lat) ** 2 + (lons[i] - lon) ** 2) for i in range(len(lats))
    ]
    ind = np.where(dist == np.min(dist))[0][0]

    logging.info(
        "Closest ind %d. Given lon-lat: %f,%f, found lon-lat: %f,%f",
        ind,
        lon,
        lat,
        lons[ind],
        lats[ind],
    )

    return ind


def parse_coords(domain: str) -> Tuple[float, float]:
    """Parse coordinates from domain string with keyword 'nn'.

    Parameters
    ----------
    domain : str
        keyword nn: followed by lon,lat, e.g 'nn:8.4,46.3'

    Returns
    -------
    lon : float
        Longitude (numerical value)
    lat : float
        Latitude (numerical value)

    """
    coords_list = domain.split(",")
    lon = float(coords_list[0])
    lat = float(coords_list[1])
    return lon, lat


def mask_domain(
    da: xr.DataArray, domain: str, grid: IconGrid | None = None
) -> xr.DataArray:
    """Apply domain mask to Dataarray."""
    # get domain
    dom_pts, _ = get_domain(domain)
    # apply the domain mask
    if da.attrs["GRIB_gridType"] == "rotated_ll":
        points = np.stack(
            [da.longitude.values.flatten(), da.latitude.values.flatten()], axis=1
        )
        mask = points_in_domain(points, dom_pts)
        mask_da = xr.DataArray(
            mask.reshape(da.longitude.values.shape),
            dims=["y", "x"],
            coords={"longitude": da["longitude"], "latitude": da["latitude"]},
            name="mask",
        )
        # apply domain mask
        da = da.where(mask_da)
    elif da.attrs["GRIB_gridType"] == "unstructured_grid":
        # mask grid points outside of the domain
        assert grid is not None
        # safety first :-)
        grid.mask_domain(dom_pts)
        mask_da = xr.DataArray(
            grid.mask, coords={"values": da.coords["values"]}, name="mask"
        )
        # apply domain mask
        da = da.where(~mask_da)  # pylint: disable=invalid-unary-operand-type
    return da


# pylint: disable=too-many-arguments, duplicate-code
def check_grid(
    filelist: List[str],
    grid: IconGrid | None,
    varname: str,
    level: int | None = None,
    chunks: Dict[str, int] | None = None,
    dask_nworkers: int | None = None,
) -> None:
    """Check if provided grid matches provided dataset."""
    logging.info("read grid file and check if grid is compatible")
    # check if gridfile is needed on first file
    da = var_from_files(
        [filelist[0]],
        varname,
        level,
        parallel=True,
        chunks=chunks,
        dask_nworkers=dask_nworkers,
    )
    if da.attrs["GRIB_gridType"] == "unstructured" and not grid:
        logging.error("the data grid is unstructured, please provide a grid file!")
        sys.exit()
    elif da.attrs["GRIB_gridType"] == "unstructured_grid" and grid:
        # check compatibility
        if not grid.check_compatibility(da):
            logging.error("grid and data are not compatible! size mismatch")
            sys.exit()
    elif da.attrs["GRIB_gridType"] not in ["unstructured_grid", "rotated_ll"]:
        logging.error(
            "no support for domain selection on grid type %s",
            da.attrs["GRIB_gridType"],
        )
        sys.exit()


# pylint: enable=too-many-arguments, duplicate-code
