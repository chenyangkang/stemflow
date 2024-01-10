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
[![codecov](https://codecov.io/gh/chenyangkang/stemflow/graph/badge.svg?token=RURPF6NKIJ)](https://codecov.io/gh/chenyangkang/stemflow)
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

-----

## Brief introduction :information_source:
**stemflow** is a toolkit for Adaptive Spatio-Temporal Exploratory Model (AdaSTEM \[[1](https://ojs.aaai.org/index.php/AAAI/article/view/8484), [2](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/eap.2056)\]) in python. A typical usage is daily abundance estimation using [eBird](https://ebird.org/home) citizen science data (survey data). 

**stemflow** adopts ["split-apply-combine"](https://vita.had.co.nz/papers/plyr.pdf) philosophy. It 

1. Splits input data using [Quadtree algorithm](https://en.wikipedia.org/wiki/Quadtree#:~:text=A%20quadtree%20is%20a%20tree,into%20four%20quadrants%20or%20regions.).
1. Trains each spatiotemporal split (called stixel) separately.
1. Aggregates the ensemble to make prediction.


The framework leverages the "adjacency" information of surroundings in space and time to model/predict the values of target spatiotemporal points. This framework ameliorates the **long-distance/long-range prediction problem** [[3](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/09-1340.1)], and have a good spatiotemporal smoothing effect.

For more information, please see [an introduction to stemflow](https://chenyangkang.github.io/stemflow/A_brief_introduction/A_brief_introduction.html) and [learning curve analysis](https://chenyangkang.github.io/stemflow/Examples/02.AdaSTEM_learning_curve_analysis.html)

-----

## Model and data  :slot_machine:


### Main functionality of `stemflow`

:white_check_mark: Spatiotemporal modeling & prediction<br>
:white_check_mark: Calculate overall feature importances<br>
:white_check_mark: Plot spatiotemporal dynamics<br>

For details see [AdaSTEM Demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html)

---

### Supported data types

:white_check_mark: All spatial indexing (CRS)<br>
:white_check_mark: All temporal indexing<br>
:white_check_mark: Spatial-only modeling<br>
:white_check_mark: Both continuous and categorical features (prefer one-hot encoding)<br>
:white_check_mark: Both static (e.g., yearly mean temperature) and dynamic features (e.g., daily temperature)

For details and tips see [Tips for data types](https://chenyangkang.github.io/stemflow/Tips/Tips_for_data_types.html)

---

### Supported tasks

:white_check_mark: Binary classification task<br>
:white_check_mark: Regression task<br>
:white_check_mark: Hurdle task (two step regression – classify then regress the non-zero part)<br>

For details and tips see [Tips for different tasks](https://chenyangkang.github.io/stemflow/Tips/Tips_for_different_tasks.html)

---

### Supported base models

:white_check_mark: sklearn style `BaseEstimator` classes ([you can make your own base model](https://scikit-learn.org/stable/developers/develop.html)), for example [here](https://chenyangkang.github.io/stemflow/Examples/06.Base_model_choices.html)<br>
:white_check_mark: sklearn style Maxent model. [Example here](https://chenyangkang.github.io/stemflow/Examples/03.Binding_with_Maxent.html).

-----

## Usage :star:

Use Hurdle model as the base model of AdaSTEMRegressor:

```py
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
from stemflow.model.Hurdle import Hurdle
from xgboost import XGBClassifier, XGBRegressor

## "hurdle in Ada"
model = AdaSTEMRegressor(
    base_model=Hurdle(
        classifier=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
        regressor=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1)
    ),                                            # hurdel model for zero-inflated problem (e.g., count)
    save_gridding_plot = True,
    ensemble_fold=10,                             # data are modeled 10 times, each time with jitter and rotation in Quadtree algo
    min_ensemble_required=7,                      # Only points covered by > 7 stixels will be predicted
    grid_len_lon_upper_threshold=25,              # force splitting if the longitudinal edge of grid exceeds 25
    grid_len_lon_lower_threshold=5,               # stop splitting if the longitudinal edge of grid fall short 5
    grid_len_lat_upper_threshold=25,              # similar to the previous one, but latitudinal
    grid_len_lat_lower_threshold=5,               
    temporal_start=1,                           # The next 4 params define the temporal sliding window
    temporal_end=366,                            
    temporal_step=20,
    temporal_bin_interval=50,
    points_lower_threshold=50,                    # Only stixels with more than 50 samples are trained
    Spatio1='longitude',                          # The next three params define the name of 
    Spatio2='latitude',                         # spatial coordinates shown in the dataframe
    Temporal1='DOY',
    use_temporal_to_train=True,                   # In each stixel, whether 'DOY' should be a predictor
    njobs=1
)
```


Fitting and prediction methods follow the style of sklearn `BaseEstimator` class:

```py
## fit
model = model.fit(X_train.reset_index(drop=True), y_train)

## predict
pred = model.predict(X_test)
pred = np.where(pred<0, 0, pred)
eval_metrics = AdaSTEM.eval_STEM_res('hurdle',y_test, pred_mean)
print(eval_metrics)
```

Where the `pred` is the mean of the predicted values across ensembles.

See [AdaSTEM demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html) for further functionality.<br>
See [Optimizing Stixel Size](https://chenyangkang.github.io/stemflow/Examples/07.Optimizing_Stixel_Size.html) for why and how you should tune the important gridding parameters.

-----

## Plot QuadTree ensembles :evergreen_tree:


```py
model.gridding_plot
# Here, the model is a AdaSTEM class, not a hurdle class
```

![QuadTree example](https://chenyangkang.github.io/stemflow/QuadTree.png)

Here, each color shows an ensemble generated during model fitting. In each of the 10 ensembles, regions (in terms of space and time) with more training samples were gridded into finer resolution, while the sparse one remained coarse. Prediction results were aggregated across the ensembles (that is, in this example, data were modeled 10 times).

----
## Example of visualization :world_map:

Daily Abundance Map of Barn Swallow

![GIF visualization](https://github.com/chenyangkang/stemflow/raw/main/docs/pred_gif.gif)

See section [Prediction and Visualization](https://chenyangkang.github.io/stemflow/Examples/04.Prediction_visualization.html) for how to generate this GIF.

----

## Contribute to stemflow :purple_heart:

We welcome pull requests. Contributors should follow [contributor guidelines](https://github.com/chenyangkang/stemflow/blob/main/docs/CONTRIBUTING.md).

Application level cooperation is also welcomed. We recognized that stemflow may consume large computational resources especially as data volume boosts in the future. We always welcome research collaboration of all kinds. Contact me at chenyangkang24@outlook.com


-----
References:

1. [Fink, D., Damoulas, T., & Dave, J. (2013, June). Adaptive Spatio-Temporal Exploratory Models: Hemisphere-wide species distributions from massively crowdsourced eBird data. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 27, No. 1, pp. 1284-1290).](https://ojs.aaai.org/index.php/AAAI/article/view/8484)

1. [Fink, D., Auer, T., Johnston, A., Ruiz‐Gutierrez, V., Hochachka, W. M., & Kelling, S. (2020). Modeling avian full annual cycle distribution and population trends with citizen science data. Ecological Applications, 30(3), e02056.](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/eap.2056)

1. [Fink, D., Hochachka, W. M., Zuckerberg, B., Winkler, D. W., Shaby, B., Munson, M. A., ... & Kelling, S. (2010). Spatiotemporal exploratory models for broad‐scale survey data. Ecological Applications, 20(8), 2131-2147.](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/09-1340.1)

1. [Johnston, A., Fink, D., Reynolds, M. D., Hochachka, W. M., Sullivan, B. L., Bruns, N. E., ... & Kelling, S. (2015). Abundance models improve spatial and temporal prioritization of conservation resources. Ecological Applications, 25(7), 1749-1756.](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/14-1826.1)
