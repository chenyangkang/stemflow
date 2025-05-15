-------
stemflow version 1.1.3
-------
**May 14, 2025:**

1. ensemble_bootstrap argument: Defaults to False. if True, the data will be bootstrapped once for each ensemble. In this case users can generate ensemble-level uncertainty, accounting for variance in data.
2. joblib_backend argument: Defaults 'loky'. Other available arguments include 'threading' ('multiprocessing' will not work with `generator` as the return). Sometimes only threading may work on certain systems.
3. base_model_method argument: defaults to None. If None, `predict` or `predict_proba` will be used depending on the tasks. This argument is handy if you have a custom base model class that has a special prediction function. Notice that dummy model will still predict 0, so the ensemble-aggregated result is still an average of zeros and your special prediction function output. Therefore, it may only make sense if your special prediction function predicts 0 as the absense/control value. Defaults to None.

Only updated for AdaSTEM and STEM, not for SphereAdaSTEM.

<br>
<br>

**Nov 20, 2024:**

Added support for:

1.  min_class_sample. 

This allows the user to specify the threshold of "not training this base model", for the classification and hurdle tasks. In the past, this is hard coded as 1, meaning that the base model is only trained if there is at least 1 sample from a different class. Now users can set it to, e.g., 3, so that a stixel with 100 data points -- 98 0s and two 1s, will not be trained (instead, a dummy model that always predict zero will be used here), and a stixel will 100 data points -- 97 0s and three 1s will be trained.

This feature can be useful if you need to do cross-validation at base model level.

2. `n_jobs` in the `split` method.

The `split` method now use the user defined n_jobs. It was previously set to 1 since the performance on multi-core seems to be off. However, with large number of ensembles it seems to be doing a good job.

3. Passing arguments to the prediction method of base model.

This can now be realized by passing base_model_prediction_param parameters when you are calling `model.predict` or `model.predict_proba`, as long as the `predict` or `predict_proba` methods of your base model accept this argument.

4. The `logit_agg` parameter.

The `logit_agg` argument in the prediction method will allows "real" probability averaging. Meaning whether to use logit aggregation for the classification task. If True, the model is averaging the probability prediction estimated by all ensembles in logit scale, and then back-tranforms it to probability scale. It's recommended to be jointly used with the CalibratedClassifierCV class in sklearn as a wrapper of the classifier to estimate the calibrated probability. If False, the output is essentially the proportion of "1s" across the related ensembles; e.g., if 100 stixels covers this spatiotemporal points, and 90% of them predict that it is a "1", then the output probability is 0.9; Therefore it would be a probability estimated by the spatiotemporal neighborhood. Default is False, but can be set to truth for "real" probability averaging.
 
Minor changes:
1. The self.rng is now set at call of `fit`, instead of initiation stage.
2. The lazy-loading dir is created upon calling `fit`, instead of initiation stage.
3. Add probability clipping to the prediction output if using `predict_proba` in classification mode. clipping to `1e-6, 1 - 1e-6`.
4. The averaging of the probability for classification task is now on logit scale, and the `mean` prediction in the output is back-transformed to probability scale. However, the std in the output will still be in logit scale!
6. The roc_auc score is now calculated with probability and y_true. Previously a 0.5 threshold was applied to obtain a binary prediction results before calculating auc.
7. Removing "try-except" in the base model training process. If you failed in the base model training, that's a problem.

<br>
<br>


-------
stemflow version 1.1.2
-------
**Oct 25, 2024:**

Related: #59; #69

1. Add a option for completely randomized grids generation (compared to equal division of the 90 degree angle).
2. Implement Lazy-loading model dictionary for saving memory; Save ensmebles of models to disk when finish training, and loaded it when used for prediction
3. Update init parameters in AdaSTEM classes, STEM classes, and SphereAdaSTEM classes. 
4. Update lazy loading documentation & example notebooks.
5. Add related pytests.
