# %% [markdown]
# # A **mini** test of stemflow
#
# Yangkang Chen<br>
# Sep 12, 2023

import os
import pickle
import time
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np

# %%
import pandas as pd
from tqdm.auto import tqdm

from stemflow.manually_testing import run_mini_test

# from stemflow.mini_test import run_mini_test

# run_mini_test(delet_tmp_files=False, speed_up_times=2)


def load_data_from_doc():
    if not os.path.exists("./tests/stemflow_mini_test"):
        os.makedirs("./tests/stemflow_mini_test")

    print(os.getcwd())
    _ = os.popen("cp ./docs/mini_data/mini_data.pkl ./tests/stemflow_mini_test", "w")
    time.sleep(1)
    print(os.listdir("./tests/stemflow_mini_test"))
    assert os.path.exists("./tests/stemflow_mini_test/mini_data.pkl")


def test_mini(
    delet_tmp_files: bool = True,
    show: bool = False,
    ensemble_models_disk_saver=False,
    ensemble_models_disk_saving_dir="./",
    speed_up_times=1,
):
    load_data_from_doc()
    run_mini_test(
        delet_tmp_files=delet_tmp_files,
        show=show,
        ensemble_models_disk_saver=ensemble_models_disk_saver,
        ensemble_models_disk_saving_dir=ensemble_models_disk_saving_dir,
        speed_up_times=2,
        tmp_dir="./tests/stemflow_mini_test",
    )
