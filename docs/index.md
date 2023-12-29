# **stemflow** :bird:
<p align="center">
  <img src="https://chenyangkang.github.io/stemflow/logo_with_words.png" alt="stemflow logo" width="600"/>
</p>
<!--  -->
<p align="center">
  <em>A Python Package for Adaptive Spatio-Temporal Exploratory Model (AdaSTEM)</em>
</p>

![GitHub](https://img.shields.io/github/license/chenyangkang/stemflow)
![Anaconda version](https://anaconda.org/conda-forge/stemflow/badges/version.svg)
![Anaconda downloads](https://anaconda.org/conda-forge/stemflow/badges/downloads.svg)
![PyPI version](https://img.shields.io/pypi/v/stemflow)
![PyPI downloads](https://img.shields.io/pypi/dm/stemflow)
![GitHub last commit](https://img.shields.io/github/last-commit/chenyangkang/stemflow)
[![status](https://joss.theoj.org/papers/50a385b3283faf346fc16484f50f6add/status.svg)](https://joss.theoj.org/papers/50a385b3283faf346fc16484f50f6add)
 <!--  -->
 
-----

## Documentation :book:
[stemflow Documentation](https://chenyangkang.github.io/stemflow/)
<!-- stemflow -->

-----


## Installation  :wrench:

```py
pip install stemflow
```

Or using conda:

```py
conda install -c conda-forge stemflow
```

## Mini Test  :test_tube:

To run an auto-mini test, call:

```py

from stemflow.mini_test import run_mini_test

run_mini_test(delet_tmp_files=False, speed_up_times=2)

```

Or, if the package was cloned from the github repo, you can run the python script:

```py

git clone https://github.com/chenyangkang/stemflow.git
cd stemflow

pip install -r requirements.txt  # install dependencies

chmod 755 setup.py
python setup.py # installation

chmod 755 mini_test.py
python mini_test.py # run the test

```

See section [Mini Test](https://chenyangkang.github.io/stemflow/Examples/00.Mini_test.html) for further illustrations of the mini test.

## Brief introduction :information_source:
**stemflow** is a toolkit for Adaptive Spatio-Temporal Exploratory Model (AdaSTEM \[[1](https://ojs.aaai.org/index.php/AAAI/article/view/8484), [2](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/eap.2056)\]) in python. A typical usage is daily abundance estimation using eBird citizen science data. It leverages the "adjacency" information of surrounding target values in space and time to predict the classes/continuous values of target spatial-temporal points.

Stemflow is positioned as a user-friendly python package to meet the need of general application of modeling spatio-temporal large datasets. Scikit-learn style object-oriented modeling pipeline enables concise model construction with compact parameterization at the user end, while the rest of the modeling procedures are carried out under the hood. Once the fitting method is called, the model class recursively splits the input training data into smaller spatio-temporal grids (called stixels) using QuadTree algorithm. For each of the stixels, a base model is trained only using data falls into that stixel. Stixels are then aggregated and constitute an ensemble. In the prediction phase, stemflow queries stixels for the input data according to their spatial and temporal index, followed by corresponding base model prediction. Finally, prediction results are aggregated across ensembles to generate robust estimations (see [Fink et al., 2013](https://ojs.aaai.org/index.php/AAAI/article/view/8484) and stemflow documentation for details).

In the demo, we use a two-step hurdle model as "base model", with XGBoostClassifier for binary occurrence modeling and XGBoostRegressor for abundance modeling. If the task is to predict abundance, there are two ways to leverage the hurdle model. First, hurdle in AdaSTEM: one can use hurdle model in each AdaSTEM (regressor) stixel; Second, AdaSTEM in hurdle: one can use AdaSTEMClassifier as the classifier of the hurdle model, and AdaSTEMRegressor as the regressor of the hurdle model. In the first case, the classifier and regressor "talk" to each other in each separate stixel (hereafter, "hurdle in Ada"); In the second case, the classifiers and regressors form two "unions" separately, and these two unions only "talk" to each other at the final combination, instead of in each stixel (hereafter, "Ada in hurdle"). In [Johnston (2015)](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/14-1826.1) the first method was used. See section [[Hurdle in AdaSTEM or AdaSTEM in hurdle?]](https://chenyangkang.github.io/stemflow/Examples/05.Hurdle_in_ada_or_ada_in_hurdle.html) for further comparisons.

User can define the size of the stixels (spatial temporal grids) in terms of space and time. Larger stixel promotes generalizability but loses precision in fine resolution; Smaller stixel may have better predictability in the exact area but reduced ability of extrapolation for points outside the stixel.

In the demo, we first split the training data using temporal sliding windows with size of 50 day of year (DOY) and step of 20 DOY (`temporal_start = 1`, `temporal_end=366`, `temporal_step=20`, `temporal_bin_interval=50`). For each temporal slice, a spatial gridding is applied, where we force the stixel to be split into smaller 1/4 pieces if the edge is larger than 25 units (measured in longitude and latitude, `grid_len_lon_upper_threshold=25`, `grid_len_lat_upper_threshold=25`), and stop splitting to prevent the edge length being chunked below 5 units (`grid_len_lon_lower_threshold=5`, `grid_len_lat_lower_threshold=5`) or containing less than 50 checklists (`points_lower_threshold=50`). See section [[Optimizing Stixel Size]](https://chenyangkang.github.io/stemflow/Examples/07.Optimizing_Stixel_Size.html) for discussion about selection gridding parameters. Model fitting is run using 4 cores (`njobs=4`).

This process is executed 10 times (`ensemble_fold = 10`), each time with random jitter and random rotation of the gridding, generating 10 ensembles. In the prediction phase, only spatial-temporal points with more than 7 (`min_ensemble_required = 7`) ensembles usable are predicted (otherwise, set as `np.nan`).


## Usage :star:

Use Hurdle model as the base model of AdaSTEMRegressor:

```py
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
from stemflow.model.Hurdle import Hurdle_for_AdaSTEM
from xgboost import XGBClassifier, XGBRegressor

## "hurdle in Ada"
model = AdaSTEMRegressor(
    base_model=Hurdle(
        classifier=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
        regressor=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1)
    ),
    save_gridding_plot = True,
    ensemble_fold=10, 
    min_ensemble_required=7,
    grid_len_lon_upper_threshold=25,
    grid_len_lon_lower_threshold=5,
    grid_len_lat_upper_threshold=25,
    grid_len_lat_lower_threshold=5,
    temporal_start = 1, 
    temporal_end =366,
    temporal_step=20,
    temporal_bin_interval = 50,
    points_lower_threshold=50,
    Spatio1='longitude',
    Spatio2 = 'latitude', 
    Temporal1 = 'DOY',
    use_temporal_to_train=True,
    njobs=1                       
)
```


Fitting and prediction methods follow the style of sklearn `BaseEstimator` class:

```py
## fit
model.fit(X_train.reset_index(drop=True), y_train)

## predict
pred = model.predict(X_test)
pred = np.where(pred<0, 0, pred)
eval_metrics = AdaSTEM.eval_STEM_res('hurdle',y_test, pred_mean)
print(eval_metrics)
```

Where the `pred` is the mean of the predicted values across ensembles.

See [AdaSTEM demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html) for further functionality.



## Plot QuadTree ensembles :evergreen_tree:


```py
model.gridding_plot
# Here, the model is a AdaSTEM class, not a hurdle class
```

![QuadTree example](https://chenyangkang.github.io/stemflow/QuadTree.png)

Here, each color shows an ensemble generated during model fitting. In each of the 10 ensembles, regions (in terms of space and time) with more training samples were gridded into finer resolution, while the sparse one remained coarse. Prediction results were aggregated across the ensembles (that is, in this example, data were gone through 10 times).

---- 
## Example of visualization :world_map:

Daily Abundance Map of Barn Swallow

![GIF visualization](https://github.com/chenyangkang/stemflow/raw/main/docs/pred_gif.gif)

See section [Prediction and Visualization](https://chenyangkang.github.io/stemflow/Examples/04.Prediction_visualization.html) for how to generate this GIF.

----

## Contribute to stemflow :purple_heart:

**Pull requests are welcomed!** Open an issue so that we can discuss the detailed implementation.

**Application level cooperation is also welcomed!** My domain knowledge is in avian ecology and evolution. 

You can contact me at **chenyangkang24@outlook.com**


-----
References:

1. [Fink, D., Damoulas, T., & Dave, J. (2013, June). Adaptive Spatio-Temporal Exploratory Models: Hemisphere-wide species distributions from massively crowdsourced eBird data. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 27, No. 1, pp. 1284-1290).](https://ojs.aaai.org/index.php/AAAI/article/view/8484)

2. [Fink, D., Auer, T., Johnston, A., Ruiz‐Gutierrez, V., Hochachka, W. M., & Kelling, S. (2020). Modeling avian full annual cycle distribution and population trends with citizen science data. Ecological Applications, 30(3), e02056.](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/eap.2056)

3. [Johnston, A., Fink, D., Reynolds, M. D., Hochachka, W. M., Sullivan, B. L., Bruns, N. E., ... & Kelling, S. (2015). Abundance models improve spatial and temporal prioritization of conservation resources. Ecological Applications, 25(7), 1749-1756.](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/14-1826.1)
