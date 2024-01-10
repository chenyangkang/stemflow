# A brief introduction to `stemflow`

**stemflow** is a toolkit for Adaptive Spatio-Temporal Exploratory Model (AdaSTEM \[[1](https://ojs.aaai.org/index.php/AAAI/article/view/8484), [2](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/eap.2056)\]) in python. A typical usage is daily abundance estimation using [eBird](https://ebird.org/home) citizen science data (survey data). 

**stemflow** adapts ["split-apply-combine"](https://vita.had.co.nz/papers/plyr.pdf) philosophy. It 

1. Splits input data using Quadtree algorithm
1. Train each spatiotemporal split (called stixel) seperately.
1. Aggregate then ensemble to make prediction.


The framework leverages the "adjacency" information of surroundings in space and time to model/predict the values of target spatiotemporal points. This framework ameliorates the **long-distance/long-range prediction problem** [[3](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/09-1340.1)], and have a good spatiotemporal smoothing effect.


Technically, stemflow is positioned as a user-friendly python package to meet the need of general application of modeling spatio-temporal large datasets. Scikit-learn style object-oriented modeling pipeline enables concise model construction with compact parameterization at the user end, while the rest of the modeling procedures are carried out under the hood. Once the fitting method is called, the model class recursively splits the input training data into smaller spatio-temporal grids (called stixels) using QuadTree algorithm. For each of the stixels, a base model is trained only using data falls into that stixel. Stixels are then aggregated and constitute an ensemble. In the prediction phase, stemflow queries stixels for the input data according to their spatial and temporal index, followed by corresponding base model prediction. Finally, prediction results are aggregated across ensembles to generate robust estimations (see [Fink et al., 2013](https://ojs.aaai.org/index.php/AAAI/article/view/8484) and stemflow documentation for details).

## Choosing the model framework

In the [demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html), we use `a two-step hurdle model` as "base model" (see more information about `hurdle` model [here](https://chenyangkang.github.io/stemflow/Tips/Tips_for_different_tasks.html)), with XGBoostClassifier for binary occurrence modeling and XGBoostRegressor for abundance modeling. If the task is to predict abundance, there are two ways to leverage the hurdle model. 

1. First, **hurdle in AdaSTEM**: one can use hurdle model in each AdaSTEM (regressor) stixel; 
1. Second, **AdaSTEM in hurdle**: one can use `AdaSTEMClassifier` as the classifier of the hurdle model, and `AdaSTEMRegressor` as the regressor of the hurdle model. 

In the first case, the classifier and regressor "talk" to each other in each separate stixel (hereafter, "hurdle in Ada"); In the second case, the classifiers and regressors form two "unions" separately, and these two unions only "talk" to each other at the final combination, instead of in each stixel (hereafter, "Ada in hurdle"). In [Johnston (2015)](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/14-1826.1) the first method was used. See section [[Hurdle in AdaSTEM or AdaSTEM in hurdle?]](https://chenyangkang.github.io/stemflow/Examples/05.Hurdle_in_ada_or_ada_in_hurdle.html) for further comparisons.

## Choosing the gird size
User can define the size of the stixels (spatial temporal grids) in terms of space and time. Larger stixel promotes generalizability but loses precision in fine resolution; Smaller stixel may have better predictability in the exact area but reduced ability of extrapolation for points outside the stixel. See section [Optimizing Stixel Size](https://chenyangkang.github.io/stemflow/Examples/07.Optimizing_Stixel_Size.html) for discussion about selection gridding parameters.

## A simple demo
In the demo, we first split the training data using temporal sliding windows with size of 50 day of year (DOY) and step of 20 DOY (`temporal_start = 1`, `temporal_end=366`, `temporal_step=20`, `temporal_bin_interval=50`). For each temporal slice, a spatial gridding is applied, where we force the stixel to be split into smaller 1/4 pieces if the edge is larger than 25 units (measured in longitude and latitude, `grid_len_lon_upper_threshold=25`, `grid_len_lat_upper_threshold=25`), and stop splitting to prevent the edge length being chunked below 5 units (`grid_len_lon_lower_threshold=5`, `grid_len_lat_lower_threshold=5`) or containing less than 50 checklists (`points_lower_threshold=50`).  Model fitting is run using 1 core (`njobs=1`).

This process is executed 10 times (`ensemble_fold = 10`), each time with random jitter and random rotation of the gridding, generating 10 ensembles. In the prediction phase, only spatial-temporal points with more than 7 (`min_ensemble_required = 7`) ensembles usable are predicted (otherwise, set as `np.nan`).

That is:

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

See [AdaSTEM demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html) for further functionality.

-----
References:

1. [Fink, D., Damoulas, T., & Dave, J. (2013, June). Adaptive Spatio-Temporal Exploratory Models: Hemisphere-wide species distributions from massively crowdsourced eBird data. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 27, No. 1, pp. 1284-1290).](https://ojs.aaai.org/index.php/AAAI/article/view/8484)

1. [Fink, D., Auer, T., Johnston, A., Ruiz‐Gutierrez, V., Hochachka, W. M., & Kelling, S. (2020). Modeling avian full annual cycle distribution and population trends with citizen science data. Ecological Applications, 30(3), e02056.](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/eap.2056)

1. [Fink, D., Hochachka, W. M., Zuckerberg, B., Winkler, D. W., Shaby, B., Munson, M. A., ... & Kelling, S. (2010). Spatiotemporal exploratory models for broad‐scale survey data. Ecological Applications, 20(8), 2131-2147.](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/09-1340.1)

1. [Johnston, A., Fink, D., Reynolds, M. D., Hochachka, W. M., Sullivan, B. L., Bruns, N. E., ... & Kelling, S. (2015). Abundance models improve spatial and temporal prioritization of conservation resources. Ecological Applications, 25(7), 1749-1756.](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1890/14-1826.1)
