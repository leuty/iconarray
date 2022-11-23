"""Handle ICON grid files."""
# Standard library
from typing import Sequence

# Third-party
import matplotlib.path as mpath  # type: ignore
import matplotlib.tri as mtri  # type: ignore
import netCDF4 as nc  # type: ignore
import numpy as np
import xarray as xr


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
            default is domain Switzerland

        Notes
        -----
        A quick performance test showed that matplotlib.path.Path.contains_point()
        is faster than shapely.geometry.polygon.Polygon.contains()

        """
        mask = np.full([self.ncells], False)
        # define closed domain
        domain_polygon = mpath.Path(shell)
        # test points
        points = np.stack([self.cx, self.cy], axis=1)
        for i, pt in enumerate(points):
            mask[i] = domain_polygon.contains_point(pt)
        # included in domain => mask inactive
        mask = ~mask

        self.set_mask(mask)


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
