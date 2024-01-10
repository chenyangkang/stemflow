# Tips for data types

In the both the [mini test](https://chenyangkang.github.io/stemflow/Examples/00.Mini_test.html) and [AdaSTEM Demo](https://chenyangkang.github.io/stemflow/Examples/01.AdaSTEM_demo.html) we use bird observation data to demonstrate functionality of AdaSTEM. Spatiotemporal coordinate are homogeneously encoded in these two cases, with `longitude` and `latitude` being spatial indexes and `DOY` (day of year) being temporal index.

Here, we present more tips and examples on how to play with these indexing systems.

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
    temporal_step=2,
    temporal_bin_interval=4,
    points_lower_threshold=50,             
    Spatio1='proj_lng',                   
    Spatio2='proj_lat',
    Temporal1='Week',
    use_temporal_to_train=True,
    njobs=1
)
```





## Continuous and categorical features

## Static and dynamic features
