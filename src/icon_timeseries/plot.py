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
        e_val = xr.DataArray()
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


# pylint: disable=too-many-arguments, too-many-locals
def plot_histograms(
    da_dict: dict[str, xr.DataArray],
    domain: str | None = None,
    min_bin: float = 0.1,
    max_bin: float = 100.0,
    nbins: int = 50,
    logbins: bool = False,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Draw a histogram plot for a dataset over a given domain."""
    logging.info("Histogram plotting started...")
    fig, axs = plt.subplots(len(da_dict), sharex=True)
    # if only one subplot, axs is not subscriptable, hence the hack.
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    # prepare bins
    # https://stackoverflow.com/questions/6855710/
    # how-to-have-logarithmic-bins-in-a-python-histogram
    if logbins:
        bins = 10.0 ** np.linspace(np.log10(min_bin), np.log10(max_bin), nbins)
    else:
        bins = np.linspace(min_bin, max_bin, nbins)

    logging.info("Number of experiments to plot: %i", len(da_dict))
    # loop over runs/experiments
    e_val = xr.DataArray()  # pylint needs to have the loop variable defined
    for i, (e_key, e_val) in enumerate(da_dict.items()):
        # Take the color sequence from a colormap
        cmap = plt.cm.get_cmap("gist_rainbow", len(e_val) + 1)
        # loop over runs/experiments
        logging.info("histogram plotting for exp %s", e_key)
        # set color
        color = cmap(i)
        vals = e_val.values.flatten()
        counts, bin_edges = np.histogram(vals, bins)
        axs[i].bar(bin_edges[:-1], counts, align="edge", color=color)
        axs[i].set_title(f"EXP {e_key}")
        axs[i].set_xlabel(f"{e_val.name} ({e_val.GRIB_units})")
        axs[i].set_ylabel("Frequency count (-)")
        axs[i].grid(True, which="both", axis="both", linestyle="--")
        if logbins:
            axs[i].set_xscale("log")

    ylims = np.array([ax.get_ylim() for ax in axs])
    ylim_max = ylims[:, 1].max()
    for ax in axs:
        ax.set_ylim(0, ylim_max)

    fig.suptitle(f"Histogram plots for domain {domain}")
    fname = f"histograms_{e_val.name}_{'-'.join(da_dict.keys())}.png"
    fig.set_size_inches(8.0, 15.0)
    fig.savefig(fname, dpi=300)
    logging.info("saved figure %s", fname)

    return fig, axs
