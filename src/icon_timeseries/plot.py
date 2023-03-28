"""Plot time series."""
# Standard library
import logging
import sys
from typing import Any
from typing import Dict
from typing import List

# Third-party
import matplotlib  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import xarray as xr
from cartopy.crs import PlateCarree  # type: ignore

# Local
from .handle_grid import IconGrid


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

    # loop over parameters
    for i, (p_key, p_val) in enumerate(da_dict.items()):
        e_val: xr.DataArray = xr.DataArray()  # pylint needs to have the loop variable
        exp_count = 0
        if colors is None:
            # Take the color sequence from a colormap
            cmap = plt.cm.get_cmap("gist_rainbow", len(p_val) + 1)
        logging.info("Number of Experiments: %i", len(p_val))

        # loop over runs/experiments
        for e_key, e_val in p_val.items():
            logging.info("plotting for parameter %s, exp %s", p_key, e_key)
            # set color
            color = cmap(exp_count)

            # get ensemble size
            try:
                len_ens = e_val.number.shape[0]
            except (
                AttributeError,
                IndexError,
            ):  # no ensemble coord or dim given or dim is 0d.
                len_ens = 1

            if len_ens > 1:
                logging.info("Looping over %i members.", e_val.number.shape[0])
                plot_ensemble(e_key, e_val, axs[i], color)
            else:
                vals = e_val.values
                time_vals = e_val.valid_time.values
                plot_ts(vals, time_vals, ax=axs[i], label=e_key, color=color)

            exp_count += 1

        # set title and legend
        title = (
            f"{e_val.name} {p_key} ({e_val.GRIB_stepType}, {e_val.GRIB_units}) "
            f"for {domain}"
        )
        if hasattr(e_val, "level"):
            title += f", level {e_val.level}"
        axs[i].set_title(title)
        axs[i].legend()

    for idx, ax in np.ndenumerate(axs):
        if idx[0] < len(axs) - 1:
            ax.xaxis.set_ticklabels([])

    if save:
        fname = f"timeseries_{e_val.name}_{'-'.join(da_dict.keys())}"
        if hasattr(e_val, "level"):
            fname += f"_l{e_val.level}"
        fname += ".png"
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
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("data time series")

    # format the time axis labeling
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=int(len(times)/5)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y %H"))
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(True, which="both", axis="both")

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
    time_vals = e_val.valid_time.values

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
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    # create figure
    fig, ax = plt.subplots(subplot_kw={"projection": PlateCarree()})

    # plot the data for the unmasked region
    try:
        cx = gd.cx[~gd.mask]
        cy = gd.cy[~gd.mask]
    except TypeError:
        cx = gd.cx
        cy = gd.cy
    vals = np.ones(cy.size)
    vals[0] = 10.0  # tricontourf does not work when all values are equal

    # plot the data
    ax, _ = _plot_map(
        cx, cy, vals, ax, transform=PlateCarree(), alpha=0.5, colormap=False
    )

    # add title
    ax.set_title(f"domain: {domain_name}")

    # save the figure
    if save:
        fname = f"domain_{domain_name.replace(' ', '')}.png"
        plt.savefig(fname, dpi=300)
        logging.info("saved figure %s", fname)

    return fig


def plot_on_map(
    data: xr.DataArray,
    gd: IconGrid,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    save: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot data on a map.

    Parameters
    ----------
    data : xr.DataArray
        data, only dimension has to be "values"
    gd : IconGrid
        grid object
    ax : matplotlib.axes.Axes, optional
        axes to use
    title : str, optional
        ax title, a default title will be generated from the data if no title is
        provided
    save : bool, optional
        save the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object
    ax : matplotlib.axes.Axes
        axes object

    """
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(subplot_kw={"projection": PlateCarree()})

    # plot the data
    ax, _ = _plot_map(gd.cx, gd.cy, data.values, ax, transform=PlateCarree())

    # set title and legend
    if title is None:
        title = f"{data.name} ({data.GRIB_stepType}, {data.GRIB_units})"
        if hasattr(data, "level"):
            title += f", level {data.level}"
    ax.set_title(title)

    # save the figure
    if save:
        fname = f"map_{data.name}"
        if hasattr(data, "level"):
            fname += f"_l{data.level}"
        fname += ".png"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        logging.info("saved figure %s", fname)

    return fig, ax


def _plot_map(
    cx: np.ndarray,
    cy: np.ndarray,
    vals: np.ndarray,
    ax: matplotlib.axes.Axes,
    colormap: bool | None = True,
    **kwargs,
) -> tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar | None]:
    """Wrap tricontourf with map background.

    Parameters
    ----------
    cx, cy : array-like
        x and y coordinates of the data
    vals : zarray-like
        height values over which the contour is drawn
    ax : matplotlib.axes.Axes
        axes object
    colormap : bool, optional
        draw a colormap
    kwargs
        keyword arguments to matplotlib.pyplot.tricontourf()

    Returns
    -------
    ax :  matplotlib.axes.Axes
        axes object
    cbar : matplotlib.colorbar.Colorbar
        colorbar object

    """
    # Third-party
    # pylint: disable=import-outside-toplevel
    from cartopy.feature import NaturalEarthFeature  # type: ignore

    # pylint: enable=import-outside-toplevel
    # plot the data
    im = ax.tricontourf(cx, cy, vals, **kwargs)
    cbar = None
    if colormap:
        cbar = plt.colorbar(im, shrink=0.5)

    # set ticklabels, suppress gridlines, set axes limits
    gl = ax.gridlines(draw_labels=True, linewidth=0)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlim(cx.min(), cx.max())
    ax.set_ylim(cy.min(), cy.max())

    # add borders and coasts
    ax.coastlines(resolution="10m", color="black")
    ax.add_feature(
        NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "10m"),
        edgecolor="black",
        facecolor="none",
    )
    return ax, cbar


# pylint: disable=too-many-arguments, too-many-locals
def plot_histograms(
    da_dict: dict[str, xr.DataArray],
    domain: str | None = None,
    min_bin: float = 0.1,
    max_bin: float = 100.0,
    nbins: int = 50,
    xlog: bool = False,
    ylog: bool = False,
    save: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Draw a histogram plot for a dataset over a given domain.

    Parameters
    ----------
    da_dict : Dict[str, xarray.DataArray]
        dictionary holding the data {"expid": xarray.DataArray}. the
        dimension of the xarray.DataArray has to be 'time'
    domain : str, optional
        name of the domain for the plot title
    min_bin : float, optional
        lowest bin bound
    max_bin : float, optional
        highest bin bound
    nbins : int, optional
        number of bins
    xlog : bool, optional
        log. x-axis
    ylog : bool, optional
        log. y-axis
    save : bool, optional
        save the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object
    ax : matplotlib.axes.Axes
        axes object

    """
    logging.info("Histogram plotting started...")
    fig, ax = plt.subplots(1)

    # prepare bins
    # https://stackoverflow.com/questions/6855710/
    # how-to-have-logarithmic-bins-in-a-python-histogram
    if xlog:
        bins = 10.0 ** np.linspace(np.log10(min_bin), np.log10(max_bin), nbins)
    else:
        bins = np.linspace(min_bin, max_bin, nbins)

    # Take the color sequence from a colormap, analogous to timeseries
    cmap = plt.cm.get_cmap("gist_rainbow", len(da_dict) + 1)

    logging.info("Number of experiments to plot: %i", len(da_dict))
    # loop over runs/experiments
    e_val: xr.DataArray = xr.DataArray()  # pylint needs to have the loop variable
    for i, (e_key, e_val) in enumerate(da_dict.items()):
        # loop over runs/experiments
        logging.info("histogram plotting for exp %s", e_key)
        # set color
        color = cmap(i)
        vals = e_val.values.flatten()
        counts, bin_edges = np.histogram(vals, bins)
        width = np.diff(bin_edges)
        ax.bar(
            bin_edges[:-1],
            counts,
            width=width,
            align="edge",
            fill=False,
            edgecolor=color,
            alpha=0.8,
            label=e_key,
        )
    ax.set_title(f"Experiments {', '.join(list(da_dict.keys()))}")  # remove brackets
    ax.set_xlabel(f"{e_val.name} {e_val.GRIB_stepType} ({e_val.GRIB_units})")
    ax.set_ylabel("Frequency count (-)")
    ax.grid(True, which="both", axis="both", linestyle="--")
    ax.legend()
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    title = f"Histogram plots for domain {domain}"
    if hasattr(e_val, "level"):
        title += f", level {e_val.level}"
    fig.suptitle(title)

    if save:
        fname = f"histograms_{e_val.name}_{'-'.join(da_dict.keys())}"
        if hasattr(e_val, "level"):
            fname += f"_l{e_val.level}"
        fname += ".png"
        # fig.set_size_inches(4.0, 8.0)
        fig.savefig(fname, dpi=300)
        logging.info("saved figure %s", fname)

    return fig, ax
