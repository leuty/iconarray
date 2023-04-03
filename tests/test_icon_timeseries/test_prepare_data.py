"""Test for data processing."""
# Standard library
import glob
import os
import pickle

# Third-party
import xarray as xr

# First-party
from icon_timeseries.prepare_data import prepare_nn

# unstructured grid
test_u = "/store/s83/cmerker/test_data/icon_timeseries/data/test_plots"
gdf_u = os.path.join(test_u, "ICON-1E_DOM01.nc")
exp_u = os.path.join(test_u, "lfff000[0-3]0000")

# regular, rotated latlon grid
test_r = "/store/s83/cmerker/test_data/icon_timeseries/data/test_rotated_ll"
exp_r = os.path.join(test_r, "2212200[0-6]_409/c1effsurf024_000")

# saved values, reference results
# update the reference with:
#   import pickle
#   f = open(pickle_file_name, "wb")
#   pickle.dump(data, f)
#   f.close()
with open(os.path.join(test_r, "test_nn_dict.pckl"), "rb") as f:
    test_nn_dict = pickle.load(f)


def test_nn_different_grids():
    """Test preparing the data for a single grid point with data on different grids."""
    # get data from test files
    da_dict = {"values": {}}
    for e, n in zip([exp_u, exp_r], ["u", "r"]):
        filelist = glob.glob(e)
        da = prepare_nn(
            filelist,
            "T_2M",
            "8.54,47.40",
            gridfile=gdf_u,
            deagg=True,
        )
        da_dict["values"][n] = da
    _ = (
        xr.testing.assert_equal(
            da_dict["values"]["u"],
            test_nn_dict["values"]["u"],
        ),
        "xarray object 'u' does not match saved reference object",
    )
    _ = (
        xr.testing.assert_equal(
            da_dict["values"]["r"],
            test_nn_dict["values"]["r"],
        ),
        "xarray object 'r' does not match saved reference object",
    )
