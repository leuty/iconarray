"""Test module ``icon_timeseries/utils.py``."""

# Third-party
import numpy as np

# First-party
from icon_timeseries.utils import ind_from_nn
from icon_timeseries.utils import nearest_xy
from icon_timeseries.utils import parse_coords


def test_nearest_xy():
    """Test nearest xy search."""
    lon2d = np.array([[3.0, 1.0], [2.0, 4.0]]).T
    lat2d = np.array([[4.0, 2.0], [3.0, 1.0]]).T
    x, y = nearest_xy(lon2d, lat2d, 1.2, 2.1)
    assert (x, y) == (0, 1)


def test_ind_from_nn():
    """Test nearest neighbour index search."""
    lons = np.array([5.0, 6.0, 7.0, 8.0])
    lats = np.array([46.0, 47.0, 48.0, 49.0])
    lon = 6.4
    lat = 48.3
    index = ind_from_nn(lons, lats, lon, lat)
    assert index == 2


def test_parse_coords():
    """Test parsing of coordinates from domain string."""
    domain_str = "8.7,46.3"
    coords = parse_coords(domain_str)
    assert coords == (8.7, 46.3)
