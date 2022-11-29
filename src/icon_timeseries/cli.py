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
from .plot import plot_ts_multiple
from .plot import prepare_data

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
@click.option("--level", default=None, type=int, help="model level index")
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
    default="no",
    type=str,
    help="deagreggation of variable: Possible are 'average', 'sum' and 'no'",
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
    level: int | None,
    color: str | None,
    gridfile: str | None,
    domain: str,
    deagg: str,
    dask_nworkers: int | None,
):  # pylint: disable=too-many-arguments
    """Read data for a variable from GRIB file(s) and plot a domain average and max."""
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
    da_dict = {"mean": {}, "max": {}}  # type: Dict[str, Dict[str, xr.DataArray]]
    for one_exp in exp:
        filelist = glob.glob(one_exp[0])
        if len(filelist) == 0:
            logging.warning("file list for %s is empty, skipping...", one_exp[0])
            continue
        da_mean, da_max = prepare_data(
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

    # plot the time series
    plot_ts_multiple(da_dict, domain=domain)


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
