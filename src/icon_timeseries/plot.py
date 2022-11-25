"""Plot time series."""
import logging
import pkg_resources
import time
import sys
import yaml
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import numpy as np
import xarray as xr
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple, Union
from typing import Sequence
from .read_grib import var_from_files
from .handle_grid import get_grid, points_in_domain


def prepare_data(
    filelist: List[str],
    varname: str,
    level: int,
    gridfile: str | None = None,
    domain: str = "all",
    chunks: Dict[str, int] | None = None,
    dask_nworkers: int | None = None,
) -> Tuple[xr.DataArray, xr.DataArray]:  # pylint: disable=too-many-arguments
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
        # user-defined domain
        with pkg_resources.resource_stream("resources", "domains.yaml") as handle:
            avail_domains = yaml.safe_load(handle.read())
        try:
            domain_pts = avail_domains[domain]["points"]
        except KeyError:
            logging.error(
                "domain '%s' is not defined. add it to "
                "src/resources/domains.yaml and reinstall the package")
            sys.exit()

        # check if gridfile is needed on first file
        da = var_from_files(
            [filelist[0]],
            varname,
            level,
            parallel=True,
            chunks=chunks,
            dask_nworkers=dask_nworkers
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
                da.attrs["GRIB_gridType"]
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
        dask_nworkers=dask_nworkers
    )
    tend = time.perf_counter()
    telapsed = tend - tstart
    logging.info("reading time elapsed: %f", telapsed)

    if domain != "all":
        # apply the domain mask
        if da.attrs["GRIB_gridType"] == "rotated_ll":
            points = np.stack(
                [da.longitude.values.flatten(), da.latitude.values.flatten()],
                axis=1
            )
            mask = points_in_domain(points, domain_pts)
            # apply domain mask
            da = da.where(mask.reshape(da.longitude.values.shape))

        elif da.attrs["GRIB_gridType"] == "unstructured_grid":
            # mask grid points outside of the domain
            gd.mask_domain(domain_pts)
            # apply domain mask
            da = da.where(~ gd.mask)

    # compute average and maximum
    if da.attrs["GRIB_gridType"] == "rotated_ll":
        da_mean = da.mean(dim=["x", "y"], skipna=True).compute()
        da_max = da.max(dim=["x", "y"], skipna=True).compute()
    else:
        da_mean = da.mean(dim="values", skipna=True).compute()
        da_max = da.max(dim="values", skipna=True).compute()

    return da_mean, da_max


def plot_mean_max(da_mean: xr.DataArray, da_max: xr.DataArray, domain: str):
    """Plot the time series of the domain average and domain maximum.

    Parameters
    ----------
    da_mean : xarray.DataArray
        domain average. the dimension has to be 'time'
    da_max : xarray.DataArray
        domain maximum. the dimension has to be 'time'
    domain : str
        name of the domain fot the plot title

    """
    _, axs = plt.subplots(2)

    plot_ts(
        da_mean,
        ax=axs[0],
        title=f"{da_mean.name} mean ({da_mean.GRIB_stepType}, {da_mean.GRIB_units}) "
              f"for {domain}, level {da_mean.level}"
    )
    plot_ts(
        da_max,
        ax=axs[1],
        title=f"{da_max.name} maximum ({da_mean.GRIB_stepType}, {da_mean.GRIB_units}) "
              f"for {domain}, level {da_max.level}"
    )

    axs[0].xaxis.set_ticklabels([])

    fname = f"timeseries_{da_mean.name}_meanmax.png"
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    logging.info(f"saved figure {fname}")


def plot_ts(
    da: xr.DataArray,
    ax: matplotlib.axes.Axes | None = None,
    save: bool = False,
    title: str | None = None,
    **kwargs: Dict[str, Any]
) -> matplotlib.figure.Figure:
    """Plot a time series.

    Parameters
    ----------x
    da : xarray.DataArray
        data. the dimension has to be 'time'
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

    # get time information
    try:
        time = da.valid_time.values
    except AttributeError:
        time = da.time.values

    # plot
    ax.plot(time, da.values, **kwargs)

    # set the title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{da.name}")

    # format the time axis labeling
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=int(len(time)/5)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y %H"))
    ax.tick_params(axis='x', labelrotation=90)

    if save:
        fname = f"timeseries_{da.name}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        logging.info(f"saved figure %s", fname)

    return fig
