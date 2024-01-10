# Tips for data types

In the both the [mini test](https://chenyangkang.github.io/stemflow/Examples/00.Mini_test.html) and [AdaSTEM Demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html) we use bird observation data to demonstrate functionality of AdaSTEM. Spatiotemporal coordinate are homogeneously encoded in these two cases, with `longitude` and `latitude` being spatial indexes and `DOY` (day of year) being temporal index.

Here, we present more tips and examples on how to play with these indexing systems.

------
## Flexible coordinate systems

`stemflow` support all types of spatial coordinate reference system (CRS) and temporal indexing (for example, week month, year, or decades). `stemflow` only support tabular point data currently. You should transform your data to desired CRS before feeding them to `stemflow`.

For example, transforming CRS:

```python
import pyproj

# Define the source and destination coordinate systems
source_crs = pyproj.CRS.from_epsg(4326)  # WGS 84 (latitude, longitude)
target_crs = pyproj.CRS.from_string("ESRI:54017")  # World Behrmann equal area projection (x, y)

# Create a transformer object
transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

# Project
data['proj_lng'], data['proj_lat'] = transformer.transform(data['lng'].values, data['lat'].values)
```

Now the projected spatial coordinate for each record is stored in `data['proj_lng']` and `data['proj_lat']`

We can then feed this data to `stemflow`:

```python

from stemflow.model.AdaSTEM import AdaSTEMClassifier
from xgboost import XGBClassifier

model = AdaSTEMClassifier(
    base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0,n_jobs=1),
    save_gridding_plot = True,
    ensemble_fold=10,                      # data are modeled 10 times, each time with jitter and rotation in Quadtree algo
    min_ensemble_required=7,               # Only points covered by > 7 stixels will be predicted
    grid_len_lon_upper_threshold=1e5,      # force splitting if the longitudinal edge of grid exceeds 1e5 meters
    grid_len_lon_lower_threshold=1e3,      # stop splitting if the longitudinal edge of grid fall short 1e3 meters
    grid_len_lat_upper_threshold=1e5,      # similar to the previous one, but latitudinal
    grid_len_lat_lower_threshold=1e3,               
    temporal_start=1,                      # The next 4 params define the temporal sliding window
    temporal_end=52,                            
    temporal_step=2,
    temporal_bin_interval=4,
    points_lower_threshold=50,             # Only stixels with more than 50 samples are trained
    Spatio1='proj_lng',                    # Use the column 'proj_lng' and 'proj_lat' as spatial indexes
    Spatio2='proj_lat',
    Temporal1='Week',
    use_temporal_to_train=True,            # In each stixel, whether 'Week' should be a predictor
    njobs=1
)
```

Here, we use temporal bin of 4 weeks and step of 2 weeks, starting from week 1 to week 52. For spatial indexing, we force the gird size to be `1km (1e3 m) ~ 10km (1e5 m)`. Since `ESRI 54017` is an equal area projection, the unit is meter.


Then we could fit the model:

```py
## fit
model = model.fit(data.drop('target', axis=1), data[['target']])

## predict
pred = model.predict(X_test)
pred = np.where(pred<0, 0, pred)
eval_metrics = AdaSTEM.eval_STEM_res('classification',y_test, pred_mean)
```

------
## Spatial-only modeling

By playing some tricks, you can also do a `spatial-only` modeling, without splitting the data into temporal blocks:

```python
model = AdaSTEMClassifier(
    base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0,n_jobs=1),
    save_gridding_plot = True,
    ensemble_fold=10,
    min_ensemble_required=7,
    grid_len_lon_upper_threshold=1e5,
    grid_len_lon_lower_threshold=1e3,
    grid_len_lat_upper_threshold=1e5,
    grid_len_lat_lower_threshold=1e3,
    temporal_start=1,
    temporal_end=52,                            
    temporal_step=1000,                 # Setting step and interval largely outweigh 
    temporal_bin_interval=1000,         # temporal scale of data
    points_lower_threshold=50,             
    Spatio1='proj_lng',                   
    Spatio2='proj_lat',
    Temporal1='Week',
    use_temporal_to_train=True,
    njobs=1
)
```

Setting `temporal_step` and `temporal_bin_interval` largely outweigh the temporal scale (1000 compared with 52) of your data will render only `one` temporal window during splitting. Consequently, your model would become a spatial model. This could be beneficial if temporal heterogeneity is not of interest, or without enough data to investigate.

------
## Continuous and categorical features

Basically, `stemflow` is a framework for spatial temporal indexing during modeling. It serves as a container to help `base model` do better jobs, and prevent distant modeling/prediction problem in space and time. Therefore, any feature you use during common tabular data modeling could be used here. It means that both continuous and categorical features can be the input, based on your expectation in the feature engineering.

For categorical features, we recommend [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) if the size of the category is not too large.

Tree-based models (e.g., decision tree, boosting tree, random forest) are robust to missing values so you can fill the missing values with artificial values like `-1`. For other methods, there are different ways to fill the missing values, with [pros and cons](https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e).

------
## Static and dynamic features

### Concepts and examples
Static features are summaries of status, while dynamic features vary largely across each almost each records.

For example, for a task modeling the abundance of a bird each day in 2020;

Static features:

    - Land cover of year 2020:
        - percentage cover of forest
        - patch density of urban area

    - Climate of year 2020:
        - BIO1: Annual Mean Temperature
        - BIO2: Mean Diurnal Range (Mean of monthly (max temp - min temp))
        - BIO19: Precipitation of Coldest Quarter

    - Normalized difference vegetation index (NDVI):
        - NDVI_max: Highest NDVI of the annual cycle
        - NDVI_std: Variation of NDVI in the annual cycle
    

Dynamic features:

    - Weather of each checklist (record):
        - the temperature of the hour (that we observed this bird)
        - the total precipitation of the hour (that we observed this bird)
        - V component of wind of the hour (that we observed this bird)

    - Normalized difference vegetation index (NDVI):
        - The absolute NDVI of the day (that we observed this bird)


### Use of static and dynamic features

Although all features except `DOY` in our [mini test](https://chenyangkang.github.io/stemflow/Examples/00.Mini_test.html) and [AdaSTEM Demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html) are static features, the model fully support dynamic feature input.

Noteworthy, the choice of static or dynamic features depends on some aspects:

1. **Model assumption**: Does the target value vary in response to static summaries or agilely in response to realtime changes?
1. **Scale of interest**: Are you interested in overall smoothed pattern or zig-zag chaotic pattern?
1. **Caution for overfitting**: `stemflow` splits data into smaller spatiotemporal grids. It may induce local overfitting to some extent. By using dynamic features, you should be additionally cautious for overfitting in the scale of time.
1. **Anchor the prediction set**: Make sure you use the same dynamic variables in your prediction set if they are used to train the model. This may cause additional computational challenges.

Likewise, we use static features for several reasons:

1. In our demonstration, static features are used as "geographical configuration". In other words, we are interested in **how birds choose different types of land according to the season**. These static features are highly summarized and have good representation for biogeographic properties.
1. We are interested in large-scale season pattern of bird migration, and are not interested in transient variation like hourly weather.
1. Keep only `DOY` as dynamic features (temporal variables) reduce the work in compiling a prediction set. Instead of making a realtime one, now we only need to change DOY (by adding one each time) and feed it to `stemflow`. It also reduces memory/IO use.

We recommend users thinking carefully before choosing appropriate features, considering the questions above and availability of computational resources.
