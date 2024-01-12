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


def run_mini_test(
    delet_tmp_files: bool = True,
    show: bool = False,
    ensemble_models_disk_saver=False,
    ensemble_models_disk_saving_dir="./",
    speed_up_times=1,
    tmp_dir="./stemflow_mini_test",
):
    """Run a mini test

    Processes:
        1. Request data
        2. ST_train_test_split
        3. Import stemflow modules
        4. Declare model instance
        5. Fitting model
        6. Calculating feature importances
        7. Assigning importance to points
        8. Plotting top 2 important variables
        9. Calculate the fitting errors
        10. Predicting on test set
        11. Evaluation
        12. Watermark
        13. Deleting tmp files (optional)

    Args:
        delet_tmp_files:
            Whether to delet files after mini test.
        show:
            Whether to show the charts in jupyter.
        ensemble_models_disk_saver:
            Whether to save ensembles in disk instead of memory.
        ensemble_models_disk_saving_dir:
            if ensemble_models_disk_saver == True, where to save the models.
        speed_up_times:
            Speed up the mini test. For example, 2 means 2 times speed up.
    """
    #
    print("Start Running Mini-test...")
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

    assert os.path.exists(len(data) > 0)

    # %%
    plt.scatter(data.longitude, data.latitude, s=0.2)
    plt.savefig(os.path.join(tmp_dir, "data_plot.pdf"))

    if show:
        plt.show()
    else:
        plt.close()

    time.sleep(1)
    assert os.path.exists(os.path.join(tmp_dir, "data_plot.pdf"))

    # %% [markdown]
    # # Get X and y

    # %%

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

    X = data.drop("count", axis=1)[x_names + ["longitude", "latitude"]]
    y = data["count"].values

    # %% [markdown]
    # # First thing first: Spatio-temporal train test split

    # %%
    import stemflow

    isl_path = stemflow.__path__[0]
    print(f"Installation path: {isl_path}")

    #
    print("ST_train_test_split ...")
    from stemflow.model_selection import ST_train_test_split

    X_train, X_test, y_train, y_test = ST_train_test_split(
        X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
    )
    print("Done.")
    assert len(X_train) > 0 and len(X_test) > 0 and len(y_train) > 0 and len(y_test) > 0

    # %% [markdown]
    # # Train AdaSTEM hurdle model

    # %%
    print("Importing stemflow modules...")
    from xgboost import XGBClassifier, XGBRegressor

    from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
    from stemflow.model.Hurdle import Hurdle, Hurdle_for_AdaSTEM

    print("Done.")

    # %%
    print("Declaring model instance...")

    fold_ = int(5 * (1 / speed_up_times))
    min_req = min([1, int(fold_ * 0.7)])
    print(f"Fold: {fold_}, min_req: {min_req}")

    model = AdaSTEMRegressor(
        base_model=Hurdle(
            classifier=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
            regressor=XGBRegressor(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        ),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_lon_upper_threshold=40,
        grid_len_lon_lower_threshold=5,
        grid_len_lat_upper_threshold=40,
        grid_len_lat_lower_threshold=5,
        temporal_start=1,
        temporal_end=366,
        temporal_step=30,
        temporal_bin_interval=60,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        use_temporal_to_train=True,
        ensemble_models_disk_saver=ensemble_models_disk_saver,
        ensemble_models_disk_saving_dir=ensemble_models_disk_saving_dir,
        njobs=1,
    )

    print("Done.")
    assert isinstance(model, AdaSTEMRegressor)

    # %%
    print("Fitting model...")
    model.fit(X_train.reset_index(drop=True), y_train, verbosity=1)
    assert len(model.model_dict) > 0
    print("Done.")

    # %% [markdown]
    # # Feature importances

    # %%
    # Calcualte feature importance. This method is automatically called when fitting the model.
    # However, to show the process, we call it again.
    print("Calculating feature importances...")
    model.calculate_feature_importances()
    assert len(model.feature_importances_) > 0

    # print(model.feature_importances_)
    # stixel-specific feature importance is saved in model.feature_importances_
    print("Done.")

    # %%
    # Assign the feature importance to spatio-temporal points of interest
    print("Assigning importance to points...")
    importances_by_points = model.assign_feature_importances_by_points(verbosity=1, njobs=1)
    assert len(importances_by_points) > 0
    print("Done.")

    # %%
    # importances_by_points.head()

    # %%
    # top 10 important variables
    top_10_important_vars = (
        importances_by_points[
            [
                i
                for i in importances_by_points.columns
                if i not in ["DOY", "longitude", "latitude", "longitude_new", "latitude_new"]
            ]
        ]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    assert len(top_10_important_vars) > 0
    print(top_10_important_vars)

    # %% [markdown]
    # ## Ploting the feature importances by variable names

    # %%
    from stemflow.utils.plot_gif import make_sample_gif

    # make spatio-temporal GIF for top 3 variables
    print("Plotting top 2 important variables...")
    for var_ in top_10_important_vars.index[:2]:
        print(f"Plotting {var_}...")
        make_sample_gif(
            importances_by_points,
            os.path.join(tmp_dir, f"FTR_IPT_{var_}.gif"),
            col=var_,
            log_scale=False,
            Spatio1="longitude",
            Spatio2="latitude",
            Temporal1="DOY",
            figsize=(18, 9),
            xlims=(data.longitude.min() - 10, data.longitude.max() + 10),
            ylims=(data.latitude.min() - 10, data.latitude.max() + 10),
            grid=True,
            xtick_interval=(data.longitude.max() - data.longitude.min()) / 8,
            ytick_interval=(data.longitude.max() - data.longitude.min()) / 8,
            lng_size=360,
            lat_size=180,
            dpi=100,
            fps=10,
        )

    assert os.path.exists(os.path.join(tmp_dir, f"FTR_IPT_{var_}.gif"))
    print("Done.")

    # %% [markdown]
    # ![GIF of feature importance for variable `slope_mean`](../FTR_IPT_slope_mean.gif)

    # %% [markdown]
    # ## Plot uncertainty (error) in training

    # %%
    # calculate mean and standard deviation in occurrence estimation
    print("Calculating the fitting errors...")
    pred_mean, pred_std = model.predict(X_train.reset_index(drop=True), return_std=True, verbosity=1, njobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0
    print("Done.")

    # %%
    # Aggregate error to hexagon
    error_df = X_train[["longitude", "latitude"]]
    error_df.columns = ["lng", "lat"]
    error_df["pred_std"] = pred_std

    # %%
    # plot error
    error_df = error_df.dropna(subset="pred_std").sample(1000)
    plt.scatter(error_df.lng, error_df.lat, c=error_df.pred_std, s=0.5)
    plt.grid(alpha=0.3)
    plt.title("Standard deviation in estimated mean occurrence")
    plt.savefig(os.path.join(tmp_dir, "error_plot.pdf"))

    if show:
        plt.show()
    else:
        plt.close()

    time.sleep(1)
    assert os.path.exists(os.path.join(tmp_dir, "error_plot.pdf"))

    # %% [markdown]
    # # Evaluation

    # %%
    print("Predicting on test set...")
    pred = model.predict(X_test)
    print("Done.")

    # %%
    perc = np.sum(np.isnan(pred.flatten())) / len(pred.flatten())
    print(f"Percentage not predictable {round(perc*100, 2)}%")
    assert perc < 0.05

    # %%
    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    # %%
    print("Evaluation...")
    eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
    print(eval)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    assert eval["Spearman_r"] >= 0.2

    print("Done.")

    # %% [markdown]
    # # Plot QuadTree ensembles

    # %%

    if show:
        model.gridding_plot.show()

    # %%
    from watermark import watermark

    print(watermark())
    print(watermark(packages="stemflow,numpy,scipy,pandas,xgboost,tqdm,matplotlib,scikit-learn,watermark"))

    # %%
    print("All Pass! ")
    if delet_tmp_files:
        print("Deleting tmp files...")
        import shutil

        shutil.rmtree(tmp_dir)
        assert not os.path.exists(tmp_dir)

    print("Finish!")
    if show:
        return model


# run_mini_test(delet_tmp_files=False, speed_up_times=2)
