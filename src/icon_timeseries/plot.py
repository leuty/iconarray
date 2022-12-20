"""Plot time series."""
# Standard library
import logging
import sys
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# Third-party
import matplotlib  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import xarray as xr

# Local
from .handle_grid import get_domain
from .handle_grid import get_grid
from .handle_grid import IconGrid
from .handle_grid import points_in_domain
from .read_grib import var_from_files
from .utils import ind_from_nn
from .utils import nearest_xy
from .utils import parse_coords


def _check_time_dim(da: xr.DataArray) -> None:
    """Check if time dimension is longer than one.

    Raise Error if shape of da.valid_time is empty tuple or 1.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray must have valid_time as an attribute (variable, coord, dim)

    """
    if da.valid_time.shape == ():
        logging.error(
            "The time dimension of the given data must be longer"
            "than one for deaggregation."
        )
        sys.exit()
    elif da.valid_time.shape[0] < 2:
        logging.error(
            "The time dimension of the given data must be longer "
            "than one for deaggregation."
        )
        sys.exit()
    else:
        pass


def deaverage(da: xr.DataArray) -> xr.DataArray:
    """Deaverage (over valid_time).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-averaged values. Needs valid_time as dimension.

    """
    try:
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    except KeyError:
        da = da.swap_dims({"time": "valid_time"})
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    subtrahend = subtrahend.assign_coords({"valid_time": da.valid_time[2:]})
    dt = da.valid_time[1] - da.valid_time[0]  # improve with unique and catch irregular
    n_fcst = ((da.valid_time[2:] - da.valid_time[0]) / dt).astype(np.int32)  # ns to h
    deavd = da
    deavd.loc[{"valid_time": da.valid_time[2:]}] = da * n_fcst - subtrahend * (
        n_fcst - 1
    )
    deavd.attrs["GRIB_stepType"] = "instant"
    return deavd


def deagg_sum(da: xr.DataArray) -> xr.DataArray:
    """Deaggregate a sum (over valid_time).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-aggregated (summed) values. Needs valid_time as dimension.

    """
    try:
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    except KeyError:
        da = da.swap_dims({"time": "valid_time"})
        _check_time_dim(da)
        subtrahend = da.sel(valid_time=da.valid_time[1:-1])
    subtrahend = subtrahend.assign_coords({"valid_time": da.valid_time[2:]})
    deaggd = da
    deaggd.loc[{"valid_time": da.valid_time[2:]}] = (
        da.loc[{"valid_time": da.valid_time[2:]}] - subtrahend
    )
    deaggd.attrs["GRIB_stepType"] = "instant"
    return deaggd


# pylint: disable=too-many-arguments
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


# pylint: enable=too-many-arguments


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


# pylint: disable=too-many-arguments
def prepare_data(
    filelist: List[str],
    varname: str,
    level: int | None = None,
    gridfile: str | None = None,
    domain: str = "all",
    deagg: str = "no",
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
    deagg : str
        Deaggregation of variable 'average', 'sum' or 'no'. default is 'no'.
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

    if deagg == "no":
        pass
    elif deagg == "average":
        da = deaverage(da)
    elif deagg == "sum":
        da = deagg_sum(da)
    else:
        raise NotImplementedError("Arg to deagg must be 'average', 'sum' or 'no'.")

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
    deagg: str = "no",
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
    deagg : str
        Deaggregation of variable 'average', 'sum' or 'no'. default is 'no'.
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

    if deagg == "no":
        pass
    elif deagg == "average":
        da = deaverage(da)
    elif deagg == "sum":
        da = deagg_sum(da)
    else:
        raise NotImplementedError("Arg to deagg must be 'average', 'sum' or 'no'.")

    lon, lat = parse_coords(lonlat)
    if "gd" in locals():  # unstructured grid
        index = ind_from_nn(gd.cx, gd.cy, lon, lat, verbose=True)
        da_nn = da.isel({"values": index})
    else:  # rotated pole
        x, y = nearest_xy(da.longitude.values, da.latitude.values, lon, lat)
        da_nn = da.isel(y=y, x=x)
    return da_nn


# pylint: enable=too-many-arguments,too-many-locals


# pylint: disable=too-many-locals
def plot_ts_multiple(
    da_dict: Dict[str, Dict[str, xr.DataArray]],
    domain: str | None = None,
    colors: List[str] | None = None,
    save: bool = True,
) -> matplotlib.figure.Figure:
    """Plot the time series of parameters defined on a domain.

    Parameters
    ----------
    da_dict : Dict[str, Dict[str, xarray.DataArray]]
        dictionary holding the data {"param": {"expid": xarray.DataArray}}. the
        dimension of the xarray.DataArray has to be 'time'
    domain : str, optional
        name of the domain for the plot title
    colors : List[str], optional
        List of (matplotlib) colors. Length must match number of experiments.
    save : bool, optional
        save the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object

    """
    fig, axs = plt.subplots(len(da_dict))
    # if only one subplot, axs is not subscriptable, hence the hack.
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, (p_key, p_val) in enumerate(da_dict.items()):
        e_val = xr.DataArray()
        exp_count = 0
        if colors is None:
            # Take the color sequence from a colormap
            cmap = plt.cm.get_cmap("gist_rainbow", len(p_val) + 1)
        logging.info("Number of Experiments: %i", len(p_val))
        for e_key, e_val in p_val.items():
            logging.info("plotting for parameter %s, exp %s", p_key, e_key)
            # set color
            color = cmap(exp_count)
            # get time information
            try:
                if e_val.number.shape[0] > 1:
                    logging.info("Looping over %i members.", e_val.number.shape[0])
                    plot_ensemble(e_key, e_val, axs[i], color)
                else:  # ensemble information given, but m is 1-dimensional
                    vals = e_val.values
                    try:
                        time_vals = e_val.valid_time.values
                    except AttributeError:
                        time_vals = e_val.time.values
                    plot_ts(vals, time_vals, ax=axs[i], label=e_key, color=color)
            except (
                AttributeError,
                IndexError,
            ):  # no ensemble coord or dim given or dim is 0d.
                vals = e_val.values
                plot_ts(vals, time_vals, ax=axs[i], label=e_key, color=color)

            exp_count += 1
        # set title and legend
        try:
            axs[i].set_title(
                f"{e_val.name} {p_key} ({e_val.GRIB_stepType}, {e_val.GRIB_units}) "
                f"for {domain}, level {e_val.level}"
            )
        except AttributeError:
            axs[i].set_title(
                f"{e_val.name} {p_key} ({e_val.GRIB_stepType}, {e_val.GRIB_units}) "
                f"for {domain}"
            )
        axs[i].legend()

    for idx, ax in np.ndenumerate(axs):
        if idx[0] < len(axs) - 1:
            ax.xaxis.set_ticklabels([])

    if save:
        fname = f"timeseries_{e_val.name}_{'-'.join(da_dict.keys())}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        logging.info("saved figure %s", fname)

    return fig


# pylint: enable=too-many-locals


def plot_ts(
    data: np.ndarray,
    times: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    save: bool = False,
    title: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Plot a time series.

    Parameters
    ----------x
    data : np.ndarray
        values
    times : np.ndarray
        valid time for values
    ax : matplotlib.axes.Axes, optional
        figure axis. use if provided, otherwise create one
    save : bool, optional
        save the figure
    title : str, optional
        title of the plot
    kwargs : Dict[Any]
        keyword arguments for matplotlib.pyplot.plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object

    """
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    # check sizes
    if not data.size == times.size:
        logging.error("dimension mismatch in data and time")
        sys.exit()

    # plot
    ax.plot(times, data, **kwargs)

    # set the title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("data time series")

    # format the time axis labeling
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=int(len(times)/5)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y %H"))
    ax.tick_params(axis="x", labelrotation=90)

    if save:
        fname = "timeseries.png"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        logging.info("saved figure %s", fname)

    return fig


def plot_ensemble(
    e_key: str, e_val: xr.DataArray, ax: matplotlib.axes.Axes, color: str
):
    """Plot a time series for each ensemble member.

    Helper function tailored to fit plot_ts_multiple and plot_ts.

    Parameter
    ---------
    e_key : str
        Experiment Key (Identifier).
    e_value :
        Experiment values in a DataArray.
    ax : matplotlib.axes.Axes
        Axes object to plot to.
    color : str
        Must be in matplotlib.colors

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object on which the time series has been plotted.

    """
    try:
        time_vals = e_val.valid_time.values
    except AttributeError:
        time_vals = e_val.time.values
    for m in e_val.number:
        vals = e_val.sel(number=m).values
        # plot
        if m.values == 0:
            plot_ts(
                vals,
                time_vals,
                ax=ax,
                label=e_key,
                alpha=1.0,
                color=color,
            )
        else:
            plot_ts(
                vals,
                time_vals,
                ax=ax,
                label="",
                alpha=0.1,
                color=color,
            )
    return ax


def plot_domain(
    gd: IconGrid,
    domain_name: str,
    ax: matplotlib.axes.Axes | None = None,
    save: bool = False,
) -> matplotlib.figure.Figure:
    """Plot a quicklook of the domain, excluding masked regions."""
    # Third-party
    # pylint: disable=import-outside-toplevel
    from cartopy.crs import PlateCarree  # type: ignore
    from cartopy.feature import NaturalEarthFeature  # type: ignore

    # pylint: enable=import-outside-toplevel

    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    # create figure
    fig, ax = plt.subplots(subplot_kw=dict(projection=PlateCarree()))

    # plot the data for the unmasked region
    try:
        cx = gd.cx[~gd.mask]
        cy = gd.cy[~gd.mask]
    except TypeError:
        cx = gd.cx
        cy = gd.cy
    vals = np.ones(cy.size)
    vals[0] = 10.0  # tricontourf does not work when all values are equal
    ax.tricontourf(cx, cy, vals, transform=PlateCarree(), alpha=0.5)

    # set ticklabels, suppress gridlines, set axes limits
    gl = ax.gridlines(draw_labels=True, linewidth=0)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlim(gd.cx.min(), gd.cx.max())
    ax.set_ylim(gd.cy.min(), gd.cy.max())

    # add borders and coasts
    ax.coastlines(resolution="10m", color="black")
    ax.add_feature(
        NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "10m"),
        edgecolor="black",
        facecolor="none",
    )

    # add title
    ax.set_title(f"domain: {domain_name}")

    # save the figute
    if save:
        fname = f"domain_{domain_name.replace(' ', '')}.png"
        plt.savefig(fname, dpi=300)
        logging.info("saved figure %s", fname)

    return fig
