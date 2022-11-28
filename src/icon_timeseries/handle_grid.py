"""Handle ICON grid files."""
# Standard library
import logging
import sys
from typing import Sequence
from typing import Tuple

# Third-party
import matplotlib.path as mpath  # type: ignore
import matplotlib.tri as mtri  # type: ignore
import netCDF4 as nc  # type: ignore
import numpy as np
import pkg_resources  # type: ignore
import xarray as xr
import yaml  # type: ignore


class IconGrid(mtri.Triangulation):
    def __init__(
        self, vx, vy, triangles, cx=None, cy=None, mask=None
    ):  # pylint: disable=too-many-arguments
        """ICON grid information class, derived from matplotlib.tri.Triangulation.

        This subclass adds the possibility to store the triangle center coordinates
        and the bounds.

        Parameters
        ----------
        vx, vy : (npoints,) array-like
            Coordinates of grid points (vertices).
        triangles : (ntri, 3) array-like of int
            For each triangle, the indices of the three points that make
            up the triangle, ordered in an anticlockwise manner.  If not
            specified, the Delaunay triangulation is calculated.
        cx, cy : (npoints,) array-like
            Coordinates of triangle centers
        mask : (ntri,) array-like of bool, optional
            Which triangles are masked out.

        Attributes
        ----------
        attrs
            see attributes of matplotlib.tri.Triangulation
        ncells : int
            number of cells
        nedges : int
            number of edges
        nverts : int
            number of vertices

        """
        mtri.Triangulation.__init__(self, vx, vy, triangles=triangles, mask=mask)
        self.cx = np.asarray(cx, dtype=np.float64)
        self.cy = np.asarray(cy, dtype=np.float64)
        self.ncells = self.cx.shape[0]
        self.negdes = self.edges.shape[0]
        self.nverts = self.x.shape[0]

    def check_compatibility(self, da: xr.DataArray) -> bool:
        """Check if grid matches data, rudimentary check on size."""
        return self.ncells == da.shape[-1]

    def mask_domain(
        self,
        shell: Sequence = [[5.75, 45.7], [5.75, 48.0], [10.8, 48.0], [10.8, 45.7]],
    ):  # pylint: disable=dangerous-default-value
        """Mask area outside a specified domain.

        Mask is True when points are masked out!

        Parameters
        ----------
        shell : sequence, optional
            a sequence of (x, y [,z]) numeric coordinate pairs or triples, or an
            array-like with shape (N, 2) or (N, 3). Also can be a sequence of Point
            objects.
            default is domain is a box around Switzerland

        """
        points = np.stack([self.cx, self.cy], axis=1)
        mask = points_in_domain(points, shell)
        # included in domain => mask inactive
        mask = ~mask

        self.set_mask(mask)


# pylint: disable=dangerous-default-value
def points_in_domain(
    points: np.ndarray,
    shell: Sequence = [[5.75, 45.7], [5.75, 48.0], [10.8, 48.0], [10.8, 45.7]],
) -> np.ndarray:
    """Test if points are contained in a domain.

    Mask is False when points are outside of the domain.

    Parameters
    ----------
    points : np.ndarray
        The points (x,y) to check. Shape (n,2)
    shell : sequence, optional
        a sequence of (x, y [,z]) numeric coordinate pairs or triples, or an
        array-like with shape (N, 2) or (N, 3). Also can be a sequence of Point
        objects.
        default is domain is a bos around Switzerland

    Returns
    -------
    mask : np.ndarray
        mask defining if points are inside (True) or outside (False) of the domain.

    Notes
    -----
    A quick performance test showed that matplotlib.path.Path.contains_point()
    is faster than shapely.geometry.polygon.Polygon.contains()

    """
    mask = np.full(points.shape[0], False)
    # define closed domain
    domain_polygon = mpath.Path(shell)
    # test points
    for i, pt in enumerate(points):
        mask[i] = domain_polygon.contains_point(pt)
    return mask


# pylint: enable=dangerous-default-value


def get_grid(gridfile: str) -> IconGrid:
    """Read an ICON grid file (netCDF) and store in a triangulation object.

    The triangulation object holds the latitude/longitude information for the
    corners of each triangle of the grid.

    Parameters
    ----------
    gridfile : str
        ICON grid file

    Returns
    -------
    gd : IconGrid
        ICON grid information

    """
    with nc.Dataset(gridfile) as f:  # pylint: disable=no-member
        # coordinates in ICON grid file are in radians
        vlon = f["vlon"][:] * 180 / np.pi
        vlat = f["vlat"][:] * 180 / np.pi
        clon = f["clon"][:] * 180 / np.pi
        clat = f["clat"][:] * 180 / np.pi
        vertex_of_cell = f["vertex_of_cell"][:]

        # vertex indices in grid file count from 1, python wants 0
        vertex_of_cell = vertex_of_cell - 1

        gd = IconGrid(vlon, vlat, vertex_of_cell.T, clon, clat)

    return gd


def get_domain(domain: str) -> Tuple[Sequence, str]:
    """Read domain information from yaml file.

    Parameters
    ----------
    domain : str
        domain to consider, please define in domains.yaml

    Returns
    -------
    dom_points : sequence
        a sequence of (x, y [,z]) numeric coordinate pairs or triples, or an
        array-like with shape (N, 2) or (N, 3). Also can be a sequence of Point
        objects.
    dom_name : str
        domain description

    """
    # user-defined domain
    with pkg_resources.resource_stream("resources", "domains.yaml") as handle:
        avail_domains = yaml.safe_load(handle.read())
    try:
        dom_pts = avail_domains[domain]["points"]
    except KeyError:
        logging.error(
            "domain '%s' is not defined. add it to "
            "src/resources/domains.yaml and reinstall the package"
        )
        sys.exit()
    try:
        dom_name = avail_domains[domain]["full_name"]
    except KeyError:
        dom_name = "undefined"

    return dom_pts, dom_name
