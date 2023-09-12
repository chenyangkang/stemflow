# stemflow
<p align="center">
  <img src="https://chenyangkang.github.io/stemflow/logo_with_words.png" alt="stemflow logo" width="600"/>
</p>
<!--  -->
<p align="center">
  <em>A package for Adaptive Spatio-Temporal Model (AdaSTEM) in python.</em>
</p>

![GitHub](https://img.shields.io/github/license/chenyangkang/stemflow)
![PyPI version](https://img.shields.io/pypi/v/stemflow)
![PyPI downloads](https://img.shields.io/pypi/dm/stemflow)
![GitHub last commit](https://img.shields.io/github/last-commit/chenyangkang/stemflow)

 <!-- ![Anaconda version](https://anaconda.org/conda-forge/stemflow/badges/version.svg) -->
 
## Installation

```py
pip install stemflow
```

## Mini Test

To run a auto-mini test, one can simply call:

```py

from stemflow.mini_test import run_mini_test

run_mini_test(delet_tmp_files=True)

```

Or, if the package were cloned from the github repo, you can run the python script:

```py
git clone https://github.com/chenyangkang/stemflow.git
cd stemflow
chmod 755 setup.py
python setup.py # installation

chmod 755 mini_test.py
python mini_test.py # run the test

```


## Brief introduction
stemflow is a toolkit for Adaptive Spatio-Temporal Model (AdaSTEM) in python. A typical usage is daily abundance estimation using eBird citizen science data. It leverages the "adjacency" information of surrounding target values in space and time, to predict the classes/continues values of target spatial-temporal point. In the demo, we use a two-step hurdle model as "base model", with XGBoostClassifier for occurence modeling and XGBoostRegressor for abundance modeling.

User can define the size of stixel (spatial temporal pixel) in terms of space and time. Larger stixel guarantees generalizability but loses precision in fine resolution; Smaller stixel may have better predictability in the exact area but reduced extrapolability for points outside the stixel.

In the demo, we first split the training data using temporal sliding windows with size of 50 day of year (DOY) and step of 20 DOY (`temporal_start = 1`, `temporal_end=366`, `temporal_step=20`, `temporal_bin_interval=50`). For each temporal slice, a spatial gridding is applied, where we force the stixel to be split into smaller 1/4 pieces if the edge is larger than 25 units (measured in longitude and latitude, `grid_len_lon_upper_threshold=25`, `grid_len_lat_upper_threshold=25`), and stop splitting to prevent the edge length to shrink below 5 units (`grid_len_lon_lower_threshold=5`, `grid_len_lat_lower_threshold=5`) or containing less than 25 checklists (`points_lower_threshold=50`).

This process is excecuted 10 times (`ensemble_fold = 10`), each time with random jitter and random rotation of the gridding, generating 10 ensembles. In the prediciton phase, only spatial-temporal points with more than 7 (`min_ensemble_required = 7`) ensembles usable are predicted (otherwise, set as `np.nan`).

Fitting and prediction methods follow the convention of sklearn `estimator` class:

```py
## fit
model.fit(X_train.reset_index(drop=True), y_train)

## predict
pred = model.predict(X_test)
pred = np.where(pred<0, 0, pred)
```

Where the pred is the mean of the predicted values across ensembles.


## Usage

```py
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
from stemflow.model.Hurdle import Hurdle_for_AdaSTEM
from xgboost import XGBClassifier, XGBRegressor

SAVE_DIR = './'


model = Hurdle_for_AdaSTEM(
    classifier=AdaSTEMClassifier(base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                                save_gridding_plot = True,
                                ensemble_fold=10, 
                                min_ensemble_required=7,
                                grid_len_lon_upper_threshold=25,
                                grid_len_lon_lower_threshold=5,
                                grid_len_lat_upper_threshold=25,
                                grid_len_lat_lower_threshold=5,
                                points_lower_threshold=50,
                                Spatio1='longitude',
                                Spatio2 = 'latitude', 
                                Temporal1 = 'DOY',
                                use_temporal_to_train=True,
                                njobs=4),
    regressor=AdaSTEMRegressor(base_model=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                                save_gridding_plot = True,
                                ensemble_fold=10, 
                                min_ensemble_required=7,
                                grid_len_lon_upper_threshold=25,
                                grid_len_lon_lower_threshold=5,
                                grid_len_lat_upper_threshold=25,
                                grid_len_lat_lower_threshold=5,
                                points_lower_threshold=50,
                                Spatio1='longitude',
                                Spatio2 = 'latitude', 
                                Temporal1 = 'DOY',
                                use_temporal_to_train=True,
                                njobs=4)
)

## fit
model.fit(X_train.reset_index(drop=True), y_train)

## predict
pred = model.predict(X_test)
pred = np.where(pred<0, 0, pred)
eval_metrics = AdaSTEM.eval_STEM_res('hurdle',y_test, pred_mean)
print(eval_metrics)

```

## Plot QuadTree ensembles


```py
model.classifier.gridding_plot
# or model.regressor.gridding_plot
```

----
![QuadTree example](https://chenyangkang.github.io/stemflow/QuadTree.png)

---- 
## Example of visualization
![GIF visualization](https://chenyangkang.github.io/stemflow/pred_gif.gif)

----
## Documentation
[stemflow Documentation](https://chenyangkang.github.io/stemflow/)
<!-- stemflow -->

-----
References:

1. [Fink, D., Damoulas, T., & Dave, J. (2013, June). Adaptive Spatio-Temporal Exploratory Models: Hemisphere-wide species distributions from massively crowdsourced eBird data. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 27, No. 1, pp. 1284-1290).](https://ojs.aaai.org/index.php/AAAI/article/view/8484)

2. [Fink, D., Auer, T., Johnston, A., Ruiz‐Gutierrez, V., Hochachka, W. M., & Kelling, S. (2020). Modeling avian full annual cycle distribution and population trends with citizen science data. Ecological Applications, 30(3), e02056.](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/eap.2056)
