"""Tests for GRIB reading."""
# Standard library
import glob
import os

# First-party
from icon_timeseries.read_grib import var_from_files

test_dir = "/store/s83/cmerker/test_data/icon_timeseries/data/test_plots"

filepattern = os.path.join(test_dir, "lfff000*0")


def test_read():
    """Test reading data from 10 files."""
    filelist = glob.glob(filepattern)
    da = var_from_files(
        filelist,
        "T",
    )
    assert da.shape == (10, 80, 1028172), "da should have shape (10, 80, 1028172)"
    assert da.GRIB_shortName == "T", "da should contain the variable T"
    assert sorted(da.dims) == [
        "generalVerticalLayer",
        "valid_time",
        "values",
    ], "da should have the dimensions 'valid_time', 'generalVerticalLayer' and 'values'"


def test_read_parallel():
    """Test reading data from 10 files, using parallel=True."""
    filelist = glob.glob(filepattern)
    da = var_from_files(
        filelist, "T", parallel=True, chunks={"generalVerticalLayer": 1}
    )
    assert da.shape == (10, 80, 1028172), "da should have shape (10, 80, 1028172)"
    assert da.GRIB_shortName == "T", "da should contain the variable T"
    assert sorted(da.dims) == [
        "generalVerticalLayer",
        "valid_time",
        "values",
    ], "da should have the dimensions 'valid_time', 'generalVerticalLayer' and 'values'"
