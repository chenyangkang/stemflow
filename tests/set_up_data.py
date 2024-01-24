import os
import pickle
from urllib.request import urlopen

import stemflow


def get_data(delet_tmp_files, tmp_dir):
    print("Temporary files will be stored at: ./stemflow_mini_test/")
    if delet_tmp_files:
        print("Temporary files will be deleted.")
    else:
        print("Temporary files will *NOT* be deleted.")

    # download mini data
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if "mini_data.pkl" not in os.listdir(tmp_dir):
        url = "https://chenyangkang.github.io/stemflow/mini_data/mini_data.pkl"
        print(f"Requesting data from {url} ...")
        data = pickle.load(urlopen(url))
        print("Done.")
    else:
        with open(os.path.join(tmp_dir, "mini_data.pkl"), "rb") as f:
            data = pickle.load(f)
        print("Mini-data already downloaded.")

    x_names = get_x_names()
    X = data.drop("count", axis=1)[x_names + ["longitude", "latitude"]]
    y = data["count"].values

    assert os.path.exists(len(data) > 0)
    return X, y


def get_x_names():
    x_names = [
        "duration_minutes",
        "Traveling",
        "DOY",
        "time_observation_started_minute_of_day",
        "elevation_mean",
        "slope_mean",
        "eastness_mean",
        "northness_mean",
        "bio1",
        "bio2",
        "bio3",
        "bio4",
        "bio5",
        "bio6",
        "bio7",
        "bio8",
        "bio9",
        "bio10",
        "bio11",
        "bio12",
        "bio13",
        "bio14",
        "bio15",
        "bio16",
        "bio17",
        "bio18",
        "bio19",
        "closed_shrublands",
        "cropland_or_natural_vegetation_mosaics",
        "croplands",
        "deciduous_broadleaf_forests",
        "deciduous_needleleaf_forests",
        "evergreen_broadleaf_forests",
        "evergreen_needleleaf_forests",
        "grasslands",
        "mixed_forests",
        "non_vegetated_lands",
        "open_shrublands",
        "permanent_wetlands",
        "savannas",
        "urban_and_built_up_lands",
        "water_bodies",
        "woody_savannas",
        "entropy",
    ]
    return x_names


def test_path():
    isl_path = stemflow.__path__[0]
    print(f"Installation path: {isl_path}")


def set_up_data(delet_tmp_files, tmp_dir):
    return get_x_names(), get_data(delet_tmp_files, tmp_dir)
