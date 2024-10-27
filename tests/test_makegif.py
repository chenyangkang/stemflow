import os
import shutil

import numpy as np
import pandas as pd

from stemflow.utils.plot_gif import make_sample_gif

size = 5000
fake_data = pd.DataFrame(
    {
        "x": np.random.uniform(low=-180, high=180, size=size),
        "y": np.random.uniform(low=-90, high=90, size=size),
        "DOY": [int(i) for i in np.random.uniform(low=1, high=366, size=size)],
        "dat": np.random.uniform(low=1, high=1000, size=size),
    }
)


def test_make_gif():
    tmp_dir = "./stemflow_test_make_gif1"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    make_sample_gif(
        fake_data,
        os.path.join(tmp_dir, "FTR_IPT_dat.gif"),
        col="dat",
        log_scale=False,
        Spatio1="x",
        Spatio2="y",
        Temporal1="DOY",
        figsize=(18, 9),
        xlims=(fake_data["x"].min() - 10, fake_data["x"].max() + 10),
        ylims=(fake_data["y"].min() - 10, fake_data["y"].max() + 10),
        grid=True,
        xtick_interval=(fake_data["x"].max() - fake_data["x"].min()) / 8,
        ytick_interval=(fake_data["x"].max() - fake_data["x"].min()) / 8,
        lng_size=360,
        lat_size=180,
        dpi=100,
        fps=10,
    )
    shutil.rmtree(tmp_dir)


def test_make_gif_changing_ranges():
    fake_data_ = fake_data[(fake_data["x"] >= 0) & (fake_data["y"] >= 10) & (fake_data["DOY"] >= 20)].copy()

    tmp_dir = "./stemflow_test_make_gif2"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    make_sample_gif(
        fake_data_,
        os.path.join(tmp_dir, "FTR_IPT_dat_changing_ranges.gif"),
        col="dat",
        log_scale=False,
        Spatio1="x",
        Spatio2="y",
        Temporal1="DOY",
        figsize=(18, 9),
        xlims=None,
        ylims=None,
        grid=True,
        xtick_interval=None,
        ytick_interval=None,
        lng_size=10,
        lat_size=10,
        dpi=100,
        fps=10,
    )

    shutil.rmtree(tmp_dir)
