import os
import shutil

import numpy as np
import pandas as pd

from stemflow.utils.plot_gif import make_sample_gif

from .make_models import make_STEMClassifier

size = 1000
fake_data = pd.DataFrame(
    {
        "x": np.random.uniform(low=-180, high=180, size=size),
        "y": np.random.uniform(low=-90, high=90, size=size),
        "DOY": np.random.uniform(low=1, high=366, size=size),
        "dat": np.random.uniform(low=1, high=1000, size=size),
    }
)


def test_make_gif():
    tmp_dir = "./stemflow_test_make_gif"
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
