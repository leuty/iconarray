# ICON time series
Visualise time series of ICON data.

This package provides functionalities to display time series from ICON output files. It handles GRIB input (necdf support is currently being developed) on the native and regular grid. The support for COSMO data should be considered beta...

## Quick start
1. Create an environment and install the dependencies: `./setup_env.sh -m`
2. Activate the environment: `conda activate icon_timeseries`
3. Install the package: `pip install .`
4. Setup ecCodes environment: `./setup_grib_env.sh`
5. Check the installation: `pytest`

You are now ready to go! For more information read below.

If the tests fail, contact the package administrator Claire Merker.

## Content
The main functionalities of this package are:
* reading
  * GRIB data (and netcdf data, soon to come)
  * unstructured and regular grids
  * ensemble data
  * multiple runs/experiments
  * optinal dask parallelisation available
* data selection
  * model variable and level to display
  * domain that can be customised
  * nearest neighbour grid point to a lon/lat location
* data processing
  * computation of average or maximum for the domain
  * de-averaging/de-accumulation of aggregated quantities
* plotting
  * time series visualisation for multiple runs
  * time series visualisation for ensemble data

The functions of the package can be used in scripts or via pre-defined command-line tools. Follwong command-line tools are provided:
* `icon-timeseries meanmax`: time series of domain average and domain maximum of a model variable on a given level (domain can be customised)
* `icon-timeseries quicklook-domain`: map plot of the considered domain
* `icon-timeseries nearest-neighbour`: time series of the values of a model variable at the grid point closest to the given location

In order to use more than one dask worker for the parallelisation, the job needs to be send to the post-proc queue. A script for `sbatch` containing an example call for `icon-timeseries meanmax` is provided: `sbatch.sh`

## Start developing

This section provides some more detailed information on the package and some guidance for the development process. Read them carefully, especially if you are new to Python and/ or APN.

### Setup

Once you created or cloned this repository, install the package dependencies with the provided script `setup_env.sh`.
Check available options with
```bash
./setup_env.sh -h
```
We distinguish development installations which are editable and have additional dependencies on formatters and linters from productive installations which are non-editable and have no additional dependencies. Moreover we distinguish pinned installations based on exported (reproducible) environments and free installations where the installation is based on first level dependencies listed in `requirements/requirements.yml` and `requirements/dev-requirements.yml` respectively. If you start developing, you might want to do a free
development installation and export the environments right away.
```bash
./setup_env.sh -d -e -m -n <package_env_name>
```
*Hint*: If you are the package administrator, it is a good idea to understand what this script does, you can do everything manually with `conda` instructions.

The package itself is installed with `pip`:
```bash
conda activate <package_env_name>
pip install --editable .
```
*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<package_env_name>/bin/pip`).

### ecCodes for GRIB decoding

Since this package uses cfgrib to decode GRIB data, make sure to run `./setup_grib_env.sh` with your conda environment active to setup ecCodes correctly. In case of an upgrade of the ecCodes versions and definitions supported by spack, this setup script might need to be updated. If you need a personalised version of ecCodes definitions that is not suported by spack, you can specify the path to your version in `GRIB_DEFINITION_PATH` (and `GRIB_SAMPLES_PATH` if needed) in `./setup_grib_env.sh`.

### Code structure

The structure of the package is currently simple:
* `src/ion_timeseries`: modules for features and utilities, and for the command-line tools (`cli.py`)
* `src/resources`: the file `domains.yaml` stores the pre-defined domains and can be extended for user-defined domains
* `tests/test_icon_timeseries`: unit and integration tests
* `requirements`: files specifying the dependencies of the package (run time: `requirements.yml`, development: `dev-requirements.yml`) and the saved pinned environments (`environment.yml` and `dev-environment.yml`)~
* `.github`: definition files for GitHub Actions workflows
* `docs`: automatically build documentation with sphinx, currently not used
* `jenkins/`: templates for Jenkins plans, currently not used

### Testing

Once your package is installed, run the tests by typing
```
pytest
```
Make sure to update your requirement files and export your environments after installationnevery time you add new imports while developing. You should add tests for every new feature you add to the package.

### Development tools

As this package was created with the APN Python blueprint, it comes with a stack of development tools, which are described in more detail on
(https://meteoswiss-apn.github.io/mch-python-blueprint/). Here, we give a brief overview on what is implemented.

#### Testing and coding standards
Testing your code and compliance with the most important Python standards is a requirement for Python software written in APN. To make the life of package
administrators easier, the most important checks are run automatically on GitHub actions. If your code goes into production, it must additionally be tested on CSCS machines, which is only possible with a Jenkins pipeline (GitHub actions is running on a GitHub server).

#### Pre-commit on GitHub actions
`.github/workflows/pre-commit.yml` contains a hook that will trigger the creation of your environment (unpinned, dev) on the GitHub actions server and
then run pytest as well as various formatters and linters through pre-commit. This hook is only triggered upon pushes to the main branch (in general: don't do that) and in pull requests to the main branch.


## Credits

This package was created with [`copier`](https://github.com/copier-org/copier) and the [`MeteoSwiss-APN/mch-python-blueprint`](https://meteoswiss-apn.github.io/mch-python-blueprint/) project template.
