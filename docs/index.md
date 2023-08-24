# Welcome to BirdSTEM

<!-- For full documentation visit [mkdocs.org](https://www.mkdocs.org). -->

## Commands

## Installation

## Fit an AdaSTEM model
```py
from BirdSTEM.model.AdaSTEM import AdaSTEM, AdaSTEMHurdle
from BirdSTEM.model.Hurdle import Hurdle
from xgboost import XGBClassifier, XGBRegressor

SAVE_DIR = './'

base_model = Hurdle(classifier=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                    regressor=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1))


model = AdaSTEMHurdle(base_model=base_model,
                        ensemble_fold = 10,
                        min_ensemble_required= 7,
                        grid_len_lon_upper_threshold=50,
                            grid_len_lon_lower_threshold=10,
                            grid_len_lat_upper_threshold=50,
                            grid_len_lat_lower_threshold=10,
                            points_lower_threshold = 50,
                            temporal_start = 0, temporal_end=1400, temporal_step=100, temporal_bin_interval = 100,
                            stixel_training_size_threshold = 50, ## important, should be consistent with points_lower_threshold
                            save_gridding_plot = True,
                            save_tmp = True,
                            save_dir=SAVE_DIR,
                            sample_weights_for_classifier=True)

## fit
model.fit(X_train,y_train)

## predict
pred_mean, pred_std = model.predict(X_test)
pred_mean = np.where(pred_mean>0, pred_mean, 0)
eval_metrics = AdaSTEM.eval_STEM_res('hurdle',y_test, pred_mean)
print(eval_metrics)

```

<!-- ## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files. -->
