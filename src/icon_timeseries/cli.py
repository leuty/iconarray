"""Command line interface of icon_timeseries."""
# Standard library
import glob
import logging
import os
import sys

# Third-party
import click

# Local
from . import __version__
from .plot import plot_mean_max
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
    "--filepattern",
    required=True,
    type=str,
    help="pattern to files to read, will be expanded to a list with glob.glob",
)
@click.option(
    "--varname", required=True, type=str, help="GRIB shortName of the variable"
)
@click.option("--level", required=True, type=int, help="model level index")
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
    "--dask-workers",
    "dask_nworkers",
    default=None,
    type=int,
    help="ignored if the script does not run on a post-proc node",
)
def avg(
    filepattern: str,
    varname: str,
    level: int,
    gridfile: str,
    domain: str,
    dask_nworkers: int | None,
):  # pylint: disable=too-many-arguments
    """Read data for a variable from GRIB file(s) and plot a domain average."""
    filelist = glob.glob(filepattern)

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

    da_mean, da_max = prepare_data(
        filelist,
        varname,
        level,
        gridfile,
        domain=domain,
        chunks=chunks,
        dask_nworkers=dask_nworkers,
    )

    # plot the time series
    plot_mean_max(da_mean, da_max, domain=domain)
