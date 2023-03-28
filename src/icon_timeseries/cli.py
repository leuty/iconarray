"""Command line interface of icon_timeseries."""
# Standard library
import glob
import logging
import os
import sys
from typing import Dict
from typing import Tuple

# Third-party
import click
import xarray as xr

# Local
from . import __version__
from .handle_grid import get_domain
from .handle_grid import get_grid
from .plot import plot_domain
from .plot import plot_histograms
from .plot import plot_on_map
from .plot import plot_ts_multiple
from .prepare_data import prepare_masked_da
from .prepare_data import prepare_meanmax
from .prepare_data import prepare_nn
from .prepare_data import prepare_time_avg
from .utils import check_grid
from .utils import datetime64_to_hourlystr as dt2str

logging.getLogger(__name__)
log_format = "%(levelname)8s: %(message)s [%(filename)s:%(lineno)s - %(funcName)s()]"
logging.basicConfig(format=log_format, level=logging.INFO, stream=sys.stdout)


# pylint: disable=W0613  # unused-argument (param)
def print_version(ctx, param, value: bool) -> None:
    """Print the version number and exit."""
    if value:
        click.echo(__version__)
        ctx.exit(0)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
@click.option(
    "--version",
    "-V",
    help="Print version and exit.",
    is_flag=True,
    expose_value=False,
    callback=print_version,
)
@click.pass_context
def main(ctx, **kwargs) -> None:
    """Console script for icon-timeseries."""
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj.update(kwargs)


@main.command()
@click.option(
    "--exp",
    required=True,
    type=(str, str),
    nargs=2,
    multiple=True,
    help=(
        "experiment info. file pattern to files to read, will be expanded to a "
        "list with glob.glob, and experiment identifier"
    ),
)
@click.option(
    "--varname", required=True, type=str, help="GRIB shortName of the variable"
)
@click.option("--level", default=None, type=float, help="model level value")
@click.option(
    "--color",
    default=None,
    type=str,
    multiple=True,
    help="color to use for experiments in plot, used as specified",
)
@click.option(
    "--gridfile",
    default=None,
    type=str,
    help="ICON grid file, needed for unstructured grid",
)
@click.option(
    "--domain",
    default="all",
    type=str,
    help="domain to consider, please define in domains.yaml",
)
@click.option(
    "--deagg",
    is_flag=True,
    help="deagreggation of variable, method is detected from GRIB encoding "
    "(de-averaging and de-accumulation are currently implemented)",
)
@click.option(
    "--dask-workers",
    "dask_nworkers",
    default=None,
    type=int,
    help="ignored if the script does not run on a post-proc node",
)
def meanmax(
    exp: Tuple[Tuple, ...],
    varname: str,
    level: float | None,
    color: str | None,
    gridfile: str | None,
    domain: str,
    deagg: bool,
    dask_nworkers: int | None,
):  # pylint: disable=too-many-arguments
    """Read data for a variable from GRIB file(s) and plot a domain average and max."""
    # check dask setup
    chunks = None
    if dask_nworkers and "ln" in os.uname().nodename:
        logging.warning(
            "job is running on %s, dask_nworkers are deactivated", os.uname().nodename
        )
        logging.warning("send your job on a post-proc node to activate dask_nworkers")
        dask_nworkers = None
    elif "ln" not in os.uname().nodename:
        logging.info("job is running on %s, dask_nworkers active", os.uname().nodename)
        logging.info("number of dask workers: %d", dask_nworkers)
        chunks = {"generalVerticalLayer": 1}

    # gather data for all experiments
    da_dict: Dict[str, Dict[str, xr.DataArray]] = {"mean": {}, "max": {}}
    for one_exp in exp:
        filelist = glob.glob(one_exp[0])
        if len(filelist) == 0:
            logging.warning("file list for %s is empty, skipping...", one_exp[0])
            continue
        da_mean, da_max = prepare_meanmax(
            filelist,
            varname,
            level,
            gridfile,
            domain=domain,
            deagg=deagg,
            chunks=chunks,
            dask_nworkers=dask_nworkers,
        )
        da_dict["mean"][one_exp[1]] = da_mean
        da_dict["max"][one_exp[1]] = da_max

    # check if any data was found
    if (len(da_dict["mean"]) == 0) or (len(da_dict["max"]) == 0):
        logging.error("No data was found.")
        sys.exit()

    # plot the time series
    plot_ts_multiple(da_dict, domain=domain)


@main.command()
@click.option(
    "--exp",
    required=True,
    type=(str, str),
    nargs=2,
    multiple=True,
    help=(
        "experiment info. file pattern to files to read, will be expanded to a "
        "list with glob.glob, and experiment identifier"
    ),
)
@click.option(
    "--varname", required=True, type=str, help="GRIB shortName of the variable"
)
@click.option(
    "--lonlat",
    required=True,
    type=str,
    help="Coordinates (format 'lon,lat') for nearest neighbour lookup.",
)
@click.option("--level", default=None, type=float, help="model level values")
@click.option(
    "--gridfile",
    default=None,
    type=str,
    help="ICON grid file, needed for unstructured grid",
)
@click.option(
    "--deagg",
    is_flag=True,
    help="deagreggation of variable, method is detected from GRIB encoding "
    "(de-averaging and de-accumulation are currently implemented)",
)
@click.option(
    "--dask-workers",
    "dask_nworkers",
    default=None,
    type=int,
    help="ignored if the script does not run on a post-proc node",
)
def nearest_neighbour(
    exp: Tuple[Tuple, ...],
    varname: str,
    level: float | None,
    gridfile: str | None,
    lonlat: str,
    deagg: bool,
    dask_nworkers: int | None,
):  # pylint: disable=too-many-arguments
    """Plot a time series from GRIB data for given variables and coordinates."""
    # check dask setup
    chunks = None
    if "pp" in os.uname().nodename:
        logging.info("job is running on %s, dask_nworkers active", os.uname().nodename)
        logging.info("number of dask workers: %d", dask_nworkers)
        chunks = {"generalVerticalLayer": 1}
    elif dask_nworkers and "pp" not in os.uname().nodename:
        logging.warning(
            "job is running on %s, dask_nworkers not active", os.uname().nodename
        )
        logging.warning("send your job on a post-proc node to activate dask_nworkers")
        dask_nworkers = None

    # gather data for all experiments
    da_dict: Dict[str, Dict[str, xr.DataArray]] = {"values": {}}
    for one_exp in exp:
        filelist = glob.glob(one_exp[0])
        if len(filelist) == 0:
            logging.warning("file list for %s is empty, skipping...", one_exp[0])
            continue
        da_point = prepare_nn(
            filelist,
            varname,
            lonlat,
            level,
            gridfile,
            deagg=deagg,
            chunks=chunks,
            dask_nworkers=dask_nworkers,
        )
        da_dict["values"][one_exp[1]] = da_point

    # plot the time series
    plot_ts_multiple(da_dict, domain=lonlat)


@main.command()
@click.option(
    "--exp",
    required=True,
    type=(str, str),
    nargs=2,
    multiple=True,
    help=(
        "experiment info. file pattern to files to read, will be expanded to a "
        "list with glob.glob, and experiment identifier"
    ),
)
@click.option(
    "--varname", required=True, type=str, help="GRIB shortName of the variable"
)
@click.option("--level", default=None, type=float, help="model level value")
@click.option(
    "--color",
    default=None,
    type=str,
    multiple=True,
    help="color to use for experiments in plot, used as specified",
)
@click.option(
    "--gridfile",
    default=None,
    type=str,
    help="ICON grid file, needed for unstructured grid",
)
@click.option(
    "--domain",
    default="all",
    type=str,
    help="domain to consider, please define in domains.yaml",
)
@click.option(
    "--deagg",
    is_flag=True,
    help="deagreggation of variable, method is detected from GRIB encoding "
    "(de-averaging and de-accumulation are currently implemented)",
)
@click.option(
    "--bins",
    type=(float, float, int),
    default=(0.1, 100.0, 50),
    nargs=3,
    help="bins for histogram, format: min (float) max (float) n_bins (int).",
)
@click.option(
    "--xlog",
    "xlog",
    default=False,
    is_flag=True,
    type=bool,
    help="plot on x-logscale with logarithmic bins",
)
@click.option(
    "--ylog",
    "ylog",
    default=False,
    is_flag=True,
    type=bool,
    help="plot on y-logscale",
)
@click.option(
    "--dask-workers",
    "dask_nworkers",
    default=None,
    type=int,
    help="ignored if the script does not run on a post-proc node",
)
def histograms(
    exp: Tuple[Tuple, ...],
    varname: str,
    level: float | None,
    color: str | None,
    gridfile: str | None,
    domain: str,
    deagg: bool,
    bins: Tuple[float, float, int],
    xlog: bool,
    ylog: bool,
    dask_nworkers: int | None,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Read data for a variable from GRIB file(s) and plot the values distribution."""
    # check dask setup
    chunks = None
    if "pp" in os.uname().nodename:
        logging.info("job is running on %s, dask_nworkers active", os.uname().nodename)
        logging.info("number of dask workers: %d", dask_nworkers)
        chunks = {"generalVerticalLayer": 1}
    elif dask_nworkers and "pp" not in os.uname().nodename:
        logging.warning(
            "job is running on %s, dask_nworkers not active", os.uname().nodename
        )
        logging.warning("send your job on a post-proc node to activate dask_nworkers")
        dask_nworkers = None

    # gather data for all experiments
    da_dict = {}  # type: Dict[str, xr.DataArray]
    for one_exp in exp:
        logging.info("Preparing experiment %s", one_exp[1])
        filelist = glob.glob(one_exp[0])
        if len(filelist) == 0:
            logging.warning("file list for %s is empty, skipping...", one_exp[0])
            continue
        da_masked = prepare_masked_da(
            filelist,
            varname,
            level,
            gridfile,
            domain=domain,
            deagg=deagg,
            chunks=chunks,
            dask_nworkers=dask_nworkers,
        )
        da_dict[one_exp[1]] = da_masked.copy()

    # plot the histograms
    _, _ = plot_histograms(
        da_dict,
        domain,
        min_bin=bins[0],
        max_bin=bins[1],
        nbins=bins[2],
        xlog=xlog,
        ylog=ylog,
    )


@main.command()
@click.option(
    "--exp",
    required=True,
    type=(str, str),
    nargs=2,
    multiple=False,
    help=(
        "experiment info. file pattern to files to read, will be expanded to a "
        "list with glob.glob, and experiment identifier"
    ),
)
@click.option(
    "--varname", required=True, type=str, help="GRIB shortName of the variable"
)
@click.option("--level", default=None, type=float, help="model level value")
@click.option(
    "--gridfile",
    required=True,
    type=str,
    help="ICON grid file, needed for unstructured grid",
)
@click.option(
    "--deagg",
    is_flag=True,
    help="deagreggation of variable, method is detected from GRIB encoding "
    "(de-averaging and de-accumulation are currently implemented)",
)
@click.option(
    "--dask-workers",
    "dask_nworkers",
    default=None,
    type=int,
    help="ignored if the script does not run on a post-proc node",
)
def time_avg(
    exp: Tuple[str, str],
    varname: str,
    level: float | None,
    gridfile: str,
    deagg: bool,
    dask_nworkers: int | None,
):  # pylint: disable=too-many-arguments,
    """Read data for variable from GRIB file(s) and plot temporally averaged field."""
    filelist = glob.glob(exp[0])
    # check dask setup
    chunks = None
    if dask_nworkers and "ln" in os.uname().nodename:
        logging.warning(
            "job is running on %s, dask_nworkers are deactivated", os.uname().nodename
        )
        logging.warning("send your job on a post-proc node to activate dask_nworkers")
        dask_nworkers = None
    elif "ln" not in os.uname().nodename:
        logging.info("job is running on %s, dask_nworkers active", os.uname().nodename)
        logging.info("number of dask workers: %d", dask_nworkers)
        chunks = {"generalVerticalLayer": 1}

    # get grid
    gd = get_grid(gridfile)
    # check compatibility of grid and data
    check_grid(filelist, gd, varname, level, chunks=chunks, dask_nworkers=dask_nworkers)

    # gather data
    if len(filelist) == 0:
        logging.warning("file list for %s is empty, nothing to do...")
        sys.exit()

    # pylint: disable=duplicate-code
    da = prepare_time_avg(
        filelist,
        varname,
        level,
        deagg=deagg,
        chunks=chunks,
        dask_nworkers=dask_nworkers,
    )
    # pylint: enable=duplicate-code

    # check dimensions
    if ("values" in da.dims) and (len(da.dims) != 1):
        logging.error(
            "The data has the wrong dimensions: %s. Please provide a level to "
            "reduce the dimensionality to 'values' only",
            da.dims,
        )
        raise ValueError("Dimensions of data for time-avg must be 'values' only.")
    if "values" not in da.dims:
        logging.error(
            "The data has the dimensions: %s. It must have 'values' only. If %s "
            "is a horizontal dimension, support must be implemented for that grid.",
            da.dims,
            da.dims,
        )
        raise NotImplementedError(
            f"Horizontal dimensions {da.dims} is not yet supported."
        )

    # plot the field
    title = f"{da.name} ({da.GRIB_stepType}, {da.GRIB_units})"
    if hasattr(da, "level"):
        title += f", level {da.level}"
    title += (
        f"\n average interval: {dt2str(da.avg_timerange[0])} - "
        f"{dt2str(da.avg_timerange[1])}"
    )

    _, _ = plot_on_map(
        da,
        gd,
        title=title,
        save=True,
    )


@main.command()
@click.option(
    "--gridfile",
    required=True,
    type=str,
    help="ICON grid file",
)
@click.option(
    "--domain",
    default="all",
    type=str,
    help="domain to consider, please define in domains.yaml",
)
def quicklook_domain(
    gridfile: str,
    domain: str,
):
    """Visualise the considered domain."""
    # read the grid
    gd = get_grid(gridfile)
    # get the domain info
    dom_pts, dom_name = get_domain(domain)
    # mask the grid
    gd.mask_domain(dom_pts)
    # plot domain
    plot_domain(gd, dom_name, save=True)
