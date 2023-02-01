# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

Please also refer to the detailed `README.md`!

## Types of Contributions

You can contribute in many ways.

### Report Bugs

Report bugs as [GitHub issues](https://github.com/MeteoSwiss-APN/icon-timeseries/issues).

If you are reporting a bug, please include

- your operating system name and version,
- any details about your local setup that might be helpful in troubleshooting, and
- detailed steps to reproduce the bug.

### Fix Bugs

Look through the [GitHub issues](https://github.com/MeteoSwiss-APN/icon-timeseries/issues) for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the  [GitHub issues](https://github.com/MeteoSwiss-APN/icon-timeseries/issues) for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

ICON time series could always use more documentation, whether as part of the official ICON time series docs or in docstrings.

### Submit Feedback

The best way to send feedback is to file a [GitHub issue]( https://github.com/MeteoSwiss-APN/icon-timeseries/issues).

If you are proposing a feature,

- explain in detail how it would work;
- keep the scope as narrow as possible, to make it easier to implement; and
- remember that this is a volunteer-driven project, and that contributions are welcome! :)

## Get Started!

Ready to contribute? Here's how to set up `icon-timeseries` for local development (see `README.md` for more detailed information).

1. Fork the [`icon-timeseries` repo](https://github.com/ on GitHub.
2. Clone your fork locally:

    ```bash
    git clone git@github.com:your_name_here/icon-timeseries.git
    ```

3. Create a virtual environment and install the dependencies:

    ```bash
    cd icon-timeseries/
    ./tools/setup_env.sh
    ```

    This will create a conda environment named `icon-timeseries` (change with `-n`) and install the pinned runtime and development dependencies in `requirements/environment.yml`.

    Setup the EcCodes environment:

    ```bash
    conda activate icon-timeseries
    tools/setup_grib_env.sh
    conda deactivate; conda activate icon-timeseries
    ```

    Install the package itself in editable mode:

    ```bash
    pip install --editable .
    ```

    Activate the environment:

    ```bash
    conda activate icon-timeseries
    ```

    Use `-u` to get the newest package versions (unpinned dependencies in `requirements/requirements.yml`), and additionally `-e` to update the environment files.

4. Create a branch for local development:

    ```bash
    git switch -c name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5. When you're done with a change, format and check the code using various installed tools like `black`, `isort`, `mypy`, `flake8` or `pylint`. Those that are set up as pre-commit hooks can be run together with:

    ```bash
    pre-commit run -a
    ```

    Next, ensure that the code does what it is supposed to do by running the tests with pytest:

    ```bash
    pytest
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "fixed this and did that"
    git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.

## Tips

For a subset of tests or a specific test, run e.g.:

```bash
pytest tests.test_icon_timeseries
pytest tests.test_icon_timeseries/test_feature::test_edge_case
```

## Versioning

In order to release a new version of your project, follow these steps:

- Make sure everything is committed, cleaned up and validating (duh!). Don't forget to keep track of the changes in `HISTORY.md`.
- Increase the version number that is hardcoded in `pyproject.toml` (and only there) and commit.
- Either create a (preferentially annotated) tag with `git tag`, or directly create a release on GitHub.
