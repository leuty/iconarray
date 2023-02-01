# ICON time series
Visualise time series of ICON data.

This package provides functionalities to display time series from ICON output files. It handles GRIB input (necdf support is currently being developed) on the native and regular grid. Support for COSMO data is implemented but the results should be checked carefully as this feature was not tested beyond a simple plausibility check.

## Quick start
1. Create an environment and install the dependencies: `./setup_env.sh -m`
2. Activate the environment: `conda activate icon_timeseries`
3. Install the package: `pip install --no-deps .`
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
  * optional dask parallelisation available
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

The functions of the package can be used in scripts or via pre-defined command-line tools. Following command-line tools are provided:
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
tools/setup_env.sh -h
```

We distinguish pinned installations based on exported (reproducible) environments and free installations where the installation
is based on top-level dependencies listed in `requirements/requirements.yml`. If you start developing, you might want to do an unpinned installation and export the environment:

```bash
tools/setup_env.sh -u -e -n <package_env_name>
```
*Hint*: If you are the package administrator, it is a good idea to understand what this script does, you can do everything manually with `conda` instructions.

*Hint*: Use the flag `-m` to speed up the installation using mamba. Of course you will have to install mamba first (we recommend to install mamba into your base
environment `conda install -c conda-forge mamba`. If you install mamba in another (maybe dedicated) environment, environments installed with mamba will be located
in `<miniconda_root_dir>/envs/mamba/envs`, which is not very practical.

The package itself is installed with `pip`. For development, install in editable mode:

```bash
conda activate <package_env_name>
pip install --editable .
```

*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<package_env_name>/bin/pip`).

Once your package is installed, run the tests by typing:

```
conda activate <package_env_name>
pytest
```

If the tests pass, you are good to go. If not, contact the package administrator Claire Merker. Make sure to update your requirement files and export your environments after installation
every time you add new imports while developing. Check the next section to find some guidance on the development process if you are new to Python and/or APN.

### ecCodes for GRIB decoding

Since this package uses cfgrib to decode GRIB data, make sure to run `./setup_grib_env.sh` with your conda environment active to setup ecCodes correctly. In case of an upgrade of the ecCodes versions and definitions supported by spack, this setup script might need to be updated. If you need a personalised version of ecCodes definitions that is not supported by spack, you can specify the path to your version in `GRIB_DEFINITION_PATH` (and `GRIB_SAMPLES_PATH` if needed) in `./setup_grib_env.sh`.

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
Make sure to update your requirement files and export your environments after installation every time you add new imports while developing. You should add tests for every new feature you add to the package.

### Roadmap to your first contribution

Tests can be triggered with `pytest` from the command line. Once you implemented a feature (and of course you also
implemented a meaningful test ;-)), you are likely willing to commit it. First, go to the root directory of your package and run pytest.

```bash
conda activate <package_env_name>
cd <package-root-dir>
pytest
```

If you use the blueprint as is, pre-commit will not be triggered locally but only if you push to the main branch
(or push to a PR to the main branch). If you consider it useful, you can set up pre-commit to run locally before every commit by initializing it once. In the root directory of
your package, type:

```bash
pre-commit install
```

If you run `pre-commit` without installing it before (line above), it will fail and the only way to recover it, is to do a forced reinstallation (`conda install --force-reinstall pre-commit`).
You can also just run pre-commit selectively, whenever you want by typing (`pre-commit run --all-files`). Note that mypy and pylint take a bit of time, so it is really
up to you, if you want to use pre-commit locally or not. In any case, after running pytest, you can commit and the linters will run at the latest on the GitHub actions server,
when you push your changes to the main branch. Note that pytest is currently not invoked by pre-commit, so it will not run automatically. Automated testing should be implemented
in a Jenkins pipeline (template for a plan available in `jenkins/`. See the next section for more details.

## Development tools

As this package was created with the APN Python blueprint, it comes with a stack of development tools, which are described in more detail on
(https://meteoswiss-apn.github.io/mch-python-blueprint/). Here, we give a brief overview on what is implemented.

### Testing and coding standards

Testing your code and compliance with the most important Python standards is a requirement for Python software written in APN. To make the life of package
administrators easier, the most important checks are run automatically on GitHub actions. If your code goes into production, it must additionally be tested on CSCS
machines, which is only possible with a Jenkins pipeline (GitHub actions is running on a GitHub server).

### Pre-commit on GitHub actions

`.github/workflows/pre-commit.yml` contains a hook that will trigger the creation of your environment (unpinned) on the GitHub actions server and
then run pytest as well as various formatters and linters through pre-commit. This hook is only triggered upon pushes to the main branch (in general: don't do that)
and in pull requests to the main branch.

### Jenkins

Two jenkins plans are available in the `jenkins/` folder. On the one hand `jenkins/Jenkinsfile` controls the nightly (weekly, monthly, ...) builds, on the other hand
`jenkins/JenkinsJobPR` controls the pipeline invoked with the command `launch jenkins` in pull requests on GitHub. Your jenkins pipeline will not be set up
automatically. If you need to run your tests on CSCS machines, contact DevOps to help you with the setup of the pipelines. Otherwise, you can ignore the jenkinsfiles
and exclusively run your tests and checks on GitHub actions.

## Features

- TODO

## Credits

This package was created with [`copier`](https://github.com/copier-org/copier) and the [`MeteoSwiss-APN/mch-python-blueprint`](https://meteoswiss-apn.github.io/mch-python-blueprint/) project template.
