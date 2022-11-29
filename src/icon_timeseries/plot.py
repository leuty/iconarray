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
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import xarray as xr

# Local
from .handle_grid import get_domain
from .handle_grid import get_grid
from .handle_grid import IconGrid
from .handle_grid import points_in_domain
from .read_grib import var_from_files


def deaverage(da: xr.DataArray) -> xr.DataArray:
    """Deaverage (over valid_time).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with time-averaged values. Needs valid_time as dimension.

    """
    try:
        subtrahend = da.sel(valid_time=da.valid_time[:-1])
    except KeyError:
        da = da.swap_dims({"time": "valid_time"})
        subtrahend = da.sel(valid_time=da.valid_time[:-1])
    subtrahend = subtrahend.assign_coords({"valid_time": da.valid_time[1:]})
    fcst_hour = ((da.valid_time[1:] - da.valid_time[0]) / 3.6e12).astype(
        np.int32
    )  # ns to h
    deavd = da
    deavd.loc[da.valid_time[1:]] = da * (fcst_hour + 1) - subtrahend * fcst_hour
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
        subtrahend = da.sel(valid_time=da.valid_time[:-1])
    except KeyError:
        da = da.swap_dims({"time": "valid_time"})
        subtrahend = da.sel(valid_time=da.valid_time[:-1])
    subtrahend = subtrahend.assign_coords({"valid_time": da.valid_time[1:]})
    deaggd = da
    deaggd.loc[da.valid_time[1:]] = da - subtrahend
    deaggd.attrs["GRIB_stepType"] = "instant"
    return deaggd


# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
# we should definitely refactor :-D
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
    if domain != "all":
        dom_pts, _ = get_domain(domain)

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
        if da.attrs["GRIB_gridType"] == "unstructured" and not gridfile:
            logging.error("the data grid is unstructured, please provide a grid file!")
            sys.exit()
        elif da.attrs["GRIB_gridType"] == "unstructured_grid" and gridfile:
            # read the grid
            gd = get_grid(gridfile)
            # check compatibility
            if not gd.check_compatibility(da):
                logging.error("grid and data are not compatible! size mismatch")
                sys.exit()
        elif da.attrs["GRIB_gridType"] not in ["unstructured_grid", "rotated_ll"]:
            logging.error(
                "no support for domain selection on grid type %s",
                da.attrs["GRIB_gridType"],
            )
            sys.exit()

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

    if domain != "all":
        # apply the domain mask
        if da.attrs["GRIB_gridType"] == "rotated_ll":
            points = np.stack(
                [da.longitude.values.flatten(), da.latitude.values.flatten()], axis=1
            )
            mask = points_in_domain(points, dom_pts)
            # apply domain mask
            da = da.where(mask.reshape(da.longitude.values.shape))

        elif da.attrs["GRIB_gridType"] == "unstructured_grid":
            # mask grid points outside of the domain
            gd.mask_domain(dom_pts)
            # apply domain mask
            da = da.where(~gd.mask)  # pylint: disable=invalid-unary-operand-type

    # compute average and maximum
    if da.attrs["GRIB_gridType"] == "rotated_ll":
        da_mean = da.mean(dim=["x", "y"], skipna=True).compute()
        da_max = da.max(dim=["x", "y"], skipna=True).compute()
    else:
        da_mean = da.mean(dim="values", skipna=True).compute()
        da_max = da.max(dim="values", skipna=True).compute()

    return da_mean, da_max


# pylint: enable=too-many-arguments, too-many-locals


def plot_ts_multiple(
    da_dict: Dict[str, Dict[str, xr.DataArray]],
    domain: str | None = None,
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
    save : bool, optional
        save the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object

    """
    fig, axs = plt.subplots(len(da_dict))

    for i, (p_key, p_val) in enumerate(da_dict.items()):
        e_val = xr.DataArray()
        for e_key, e_val in p_val.items():
            logging.info("plotting for parameter %s, exp %s", p_key, e_key)
            # get values and time information
            vals = e_val.values
            try:
                time_vals = e_val.valid_time.values
            except AttributeError:
                time_vals = e_val.time.values
            # plot
            plot_ts(vals, time_vals, ax=axs[i], label=e_key)
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

    axs[0].xaxis.set_ticklabels([])

    if save:
        fname = f"timeseries_{e_val.name}_{'-'.join(da_dict.keys())}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        logging.info("saved figure %s", fname)

    return fig


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
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y %H"))
    ax.tick_params(axis="x", labelrotation=90)

    if save:
        fname = "timeseries.png"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        logging.info("saved figure %s", fname)

    return fig


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
