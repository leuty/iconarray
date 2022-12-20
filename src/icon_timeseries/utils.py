"""Utils for the command line tool."""
# Standard library
from typing import Tuple

# Third-party
import numpy as np


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


def ind_from_nn(lats, lons, lat, lon, verbose=False):
    """Find the nearest neighbouring index to given location.

    Copy-pasted from plot_profile/plot_icon/get_icon.py (sawrry ...)
    Args:
        lats (2d array):            Latitude grid
        lons (2d array):            Longitude grid
        lat (float):                Latitude of location
        lon (float):                Longitude of location
        verbose (bool, optional):   Print information. Defaults to False.

    Returns:
        int     Index of nearest grid point.

    """
    dist = [
        np.sqrt((lats[i] - lat) ** 2 + (lons[i] - lon) ** 2) for i in range(len(lats))
    ]
    ind = np.where(dist == np.min(dist))[0][0]

    if verbose:
        print(f"Closest ind: {ind}")
        print(f" Given lat: {lat:.3f} vs found lat: {lats[ind]:.3f}")
        print(f" Given lat: {lon:.3f} vs found lon: {lons[ind]:.3f}")

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
