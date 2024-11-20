from typing import Tuple, Union

from sklearn.base import BaseEstimator

from .AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor

#


class STEM(AdaSTEM):
    """A STEM model class (allow fixed grid size)

    Parents:
        stemflow.model.AdaSTEM.AdaSTEM

    Children:
        None

    """

    def __init__(
        self,
        base_model: BaseEstimator,
        task: str = "hurdle",
        ensemble_fold: int = 10,
        min_ensemble_required: int = 7,
        grid_len: Union[float, int] = 25,
        points_lower_threshold: int = 50,
        stixel_training_size_threshold: int = None,
        temporal_start: Union[float, int] = 1,
        temporal_end: Union[float, int] = 366,
        temporal_step: Union[float, int] = 20,
        temporal_bin_interval: Union[float, int] = 50,
        temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
        spatio_bin_jitter_magnitude: Union[float, int] = "adaptive",
        random_state=None,
        save_gridding_plot: bool = True,
        sample_weights_for_classifier: bool = True,
        Spatio1: str = "longitude",
        Spatio2: str = "latitude",
        Temporal1: str = "DOY",
        use_temporal_to_train: bool = True,
        n_jobs: int = 1,
        subset_x_names: bool = False,
        plot_xlims: Tuple[Union[float, int], Union[float, int]] = None,
        plot_ylims: Tuple[Union[float, int], Union[float, int]] = None,
        verbosity: int = 0,
        plot_empty: bool = False,
        completely_random_rotation: bool = False,
        lazy_loading: bool = False,
        lazy_loading_dir: Union[str, None] = None,
        min_class_sample: int = 1
    ):
        """Make a STEM object

        Args:
            base_model:
                base model estimator
            task:
                task of the model. One of 'classifier', 'regressor' and 'hurdle'. Defaults to 'hurdle'.
            ensemble_fold:
                Ensembles count. Higher, better for the model performance. Time complexity O(N). Defaults to 10.
            min_ensemble_required:
                Only points with more than this number of model ensembles available are predicted.
                In the training phase, if stixels contain less than `points_lower_threshold` of data records,
                the results are set to np.nan, making them `unpredictable`. Defaults to 7.
            grid_len:
                length of the grids. Defaults to 25.
            points_lower_threshold:
                Do not further split the gird if split results in less samples than this threshold.
                Overriden by grid_len_*_upper_threshold parameters. Defaults to 50.
            stixel_training_size_threshold:
                Do not train the model if the available data records for this stixel is less than this threshold,
                and directly set the value to np.nan. Defaults to 50.
            temporal_start:
                start of the temporal sequence. Defaults to 1.
            temporal_end:
                end of the temporal sequence. Defaults to 366.
            temporal_step:
                step of the sliding window. Defaults to 20.
            temporal_bin_interval:
                size of the sliding window. Defaults to 50.
            temporal_bin_start_jitter:
                jitter of the start of the sliding window.
                If 'adaptive', a random jitter of range (-bin_interval, 0) will be generated
                for the start. Defaults to 'adaptive'.
            spatio_bin_jitter_magnitude:
                jitter of the spatial gridding. Defaults to 'adaptive.
            random_state:
                None or int. After setting the same seed, the model will generate the same results each time. For reproducibility.
            save_gridding_plot:
                Whether ot save gridding plots. Defaults to True.
            sample_weights_for_classifier:
                Whether to adjust for unbanlanced data for the classifier. Default to True.
            Spatio1:
                Spatial column name 1 in data. Defaults to 'longitude'.
            Spatio2:
                Spatial column name 2 in data. Defaults to 'latitude'.
            Temporal1:
                Temporal column name 1 in data.  Defaults to 'DOY'.
            use_temporal_to_train:
                Whether to use temporal variable to train. For example in modeling the daily abundance of bird population,
                whether use 'day of year (DOY)' as a training variable. Defaults to True.
            n_jobs:
                Number of multiprocessing in fitting the model. Defaults to 1.
            subset_x_names:
                Whether to only store variables with std > 0 for each stixel. Set to False will significantly increase the training speed.
            plot_xlims:
                If save_gridding_plot=true, what is the xlims of the plot. Defaults to the extent of input X varibale.
            plot_ylims:
                If save_gridding_plot=true, what is the ylims of the plot. Defaults to the extent of input Y varibale.
            verbosity:
                0 to output nothing and everything otherwise.
            plot_empty:
                Whether to plot the empty grid.
            completely_random_rotation:
                If True, the rotation angle will be generated completely randomly, as in paper https://doi.org/10.1002/eap.2056. If False, the ensembles will split the 90 degree with equal angle intervals. e.g., if ensemble_fold=9, then each ensemble will rotate 10 degree futher than the previous ensemble. Defalt to False, because if ensemble fold is small, it will be more robust to equally devide the data; and if ensemble fold is large, they are effectively similar than complete random.
            lazy_loading:
                If True, ensembles of models will be saved in disk, and only loaded when being used (e.g., prediction phase), and the ensembles of models are dump to disk once it is used.
            lazy_loading_dir:
                If lazy_loading, the directory of the model to temporary save to. Default to None, where a random number will be generated as folder name.
            min_class_sample:
                Minimum umber of samples needed to train the classifier in each stixel. If the sample does not satisfy, fit a dummy one. This parameter does not influence regression tasks.

        Raises:
            AttributeError: Base model do not have method 'fit' or 'predict'
            AttributeError: task not in one of ['regression', 'classification', 'hurdle']
            AttributeError: temporal_bin_start_jitter not in one of [str, float, int]
            AttributeError: temporal_bin_start_jitter is type str, but not 'random'

        Attributes:
            x_names (list):
                All training variables used.
            stixel_specific_x_names (dict):
                stixel specific x_names (predictor variable names) for each stixel.
                We remove the variables that have no variation for each stixel.
                Therefore, the x_names are different for each stixel.
            ensemble_df (pd.core.frame.DataFrame):
                A dataframe storing the stixel gridding information.
            gridding_plot (matplotlib.figure.Figure):
                Ensemble plot.
            model_dict (dict):
                Dictionary of {stixel_index: trained_model}.
            grid_dict (dict):
                An array of stixels assigned to each ensemble.
            feature_importances_ (pd.core.frame.DataFrame):
                feature importance dataframe for each stixel.

        """
        # Init parent class
        super().__init__(
            base_model=base_model,
            task=task,
            ensemble_fold=ensemble_fold,
            min_ensemble_required=min_ensemble_required,
            grid_len_upper_threshold=None,
            grid_len_lower_threshold=None,
            points_lower_threshold=points_lower_threshold,
            stixel_training_size_threshold=stixel_training_size_threshold,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_step=temporal_step,
            temporal_bin_interval=temporal_bin_interval,
            temporal_bin_start_jitter=temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            random_state=random_state,
            save_gridding_plot=save_gridding_plot,
            sample_weights_for_classifier=sample_weights_for_classifier,
            Spatio1=Spatio1,
            Spatio2=Spatio2,
            Temporal1=Temporal1,
            use_temporal_to_train=use_temporal_to_train,
            n_jobs=n_jobs,
            subset_x_names=subset_x_names,
            plot_xlims=plot_xlims,
            plot_ylims=plot_ylims,
            verbosity=verbosity,
            plot_empty=plot_empty,
            completely_random_rotation=completely_random_rotation,
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample
        )

        self.grid_len = grid_len


class STEMClassifier(AdaSTEMClassifier):
    """STEM model Classifier interface (allow fixed grid size)

    Parents:
        stemflow.model.AdaSTEM.AdaSTEMClassifier

    Children:
        None

    Example:
        ```
        >>> from stemflow.model.STEM import STEMClassifier
        >>> from xgboost import XGBClassifier
        >>> model = STEMClassifier(base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                                save_gridding_plot = True,
                                ensemble_fold=10,
                                min_ensemble_required=7,
                                grid_len=25,
                                points_lower_threshold=50,
                                Spatio1='longitude',
                                Spatio2 = 'latitude',
                                Temporal1 = 'DOY',
                                use_temporal_to_train=True)
        >>> model.fit(X_train, y_train)
        >>> pred = model.predict(X_test)
        ```

    """

    def __init__(
        self,
        base_model: BaseEstimator,
        task: str = "classification",
        ensemble_fold: int = 10,
        min_ensemble_required: int = 7,
        grid_len: Union[float, int] = 25,
        points_lower_threshold: int = 50,
        stixel_training_size_threshold: int = None,
        temporal_start: Union[float, int] = 1,
        temporal_end: Union[float, int] = 366,
        temporal_step: Union[float, int] = 20,
        temporal_bin_interval: Union[float, int] = 50,
        temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
        spatio_bin_jitter_magnitude: Union[float, int] = "adaptive",
        random_state=None,
        save_gridding_plot: bool = True,
        sample_weights_for_classifier: bool = True,
        Spatio1: str = "longitude",
        Spatio2: str = "latitude",
        Temporal1: str = "DOY",
        use_temporal_to_train: bool = True,
        n_jobs: int = 1,
        subset_x_names: bool = False,
        plot_xlims: Tuple[Union[float, int], Union[float, int]] = None,
        plot_ylims: Tuple[Union[float, int], Union[float, int]] = None,
        verbosity: int = 0,
        plot_empty: bool = False,
        completely_random_rotation: bool = False,
        lazy_loading: bool = False,
        lazy_loading_dir: Union[str, None] = None,
        min_class_sample: int = 1
    ):
        super().__init__(
            base_model=base_model,
            task=task,
            ensemble_fold=ensemble_fold,
            min_ensemble_required=min_ensemble_required,
            grid_len_upper_threshold=None,
            grid_len_lower_threshold=None,
            points_lower_threshold=points_lower_threshold,
            stixel_training_size_threshold=stixel_training_size_threshold,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_step=temporal_step,
            temporal_bin_interval=temporal_bin_interval,
            temporal_bin_start_jitter=temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            random_state=random_state,
            save_gridding_plot=save_gridding_plot,
            sample_weights_for_classifier=sample_weights_for_classifier,
            Spatio1=Spatio1,
            Spatio2=Spatio2,
            Temporal1=Temporal1,
            use_temporal_to_train=use_temporal_to_train,
            n_jobs=n_jobs,
            subset_x_names=subset_x_names,
            plot_xlims=plot_xlims,
            plot_ylims=plot_ylims,
            verbosity=verbosity,
            plot_empty=plot_empty,
            completely_random_rotation=completely_random_rotation,
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample
        )

        self.grid_len = grid_len


class STEMRegressor(AdaSTEMRegressor):
    """STEM model Regressor interface (allow fixed grid size)

    Parents:
        stemflow.model.AdaSTEM.AdaSTEMRegressor

    Children:
        None

    Example:
    ```
    >>> from stemflow.model.STEM import STEMRegressor
    >>> from xgboost import XGBRegressor
    >>> model = STEMRegressor(base_model=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                            save_gridding_plot = True,
                            ensemble_fold=10,
                            min_ensemble_required=7,
                            grid_len=25,
                            points_lower_threshold=50,
                            Spatio1='longitude',
                            Spatio2 = 'latitude',
                            Temporal1 = 'DOY',
                            use_temporal_to_train=True)
    >>> model.fit(X_train, y_train)
    >>> pred = model.predict(X_test)
    ```

    """

    def __init__(
        self,
        base_model: BaseEstimator,
        task: str = "regression",
        ensemble_fold: int = 10,
        min_ensemble_required: int = 7,
        grid_len: Union[float, int] = 25,
        points_lower_threshold: int = 50,
        stixel_training_size_threshold: int = None,
        temporal_start: Union[float, int] = 1,
        temporal_end: Union[float, int] = 366,
        temporal_step: Union[float, int] = 20,
        temporal_bin_interval: Union[float, int] = 50,
        temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
        spatio_bin_jitter_magnitude: Union[float, int] = "adaptive",
        random_state=None,
        save_gridding_plot: bool = True,
        sample_weights_for_classifier: bool = True,
        Spatio1: str = "longitude",
        Spatio2: str = "latitude",
        Temporal1: str = "DOY",
        use_temporal_to_train: bool = True,
        n_jobs: int = 1,
        subset_x_names: bool = False,
        plot_xlims: Tuple[Union[float, int], Union[float, int]] = None,
        plot_ylims: Tuple[Union[float, int], Union[float, int]] = None,
        verbosity: int = 0,
        plot_empty: bool = False,
        completely_random_rotation: bool = False,
        lazy_loading: bool = False,
        lazy_loading_dir: Union[str, None] = None,
        min_class_sample: int = 1
    ):
        super().__init__(
            base_model=base_model,
            task=task,
            ensemble_fold=ensemble_fold,
            min_ensemble_required=min_ensemble_required,
            grid_len_upper_threshold=None,
            grid_len_lower_threshold=None,
            points_lower_threshold=points_lower_threshold,
            stixel_training_size_threshold=stixel_training_size_threshold,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_step=temporal_step,
            temporal_bin_interval=temporal_bin_interval,
            temporal_bin_start_jitter=temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            random_state=random_state,
            save_gridding_plot=save_gridding_plot,
            sample_weights_for_classifier=sample_weights_for_classifier,
            Spatio1=Spatio1,
            Spatio2=Spatio2,
            Temporal1=Temporal1,
            use_temporal_to_train=use_temporal_to_train,
            n_jobs=n_jobs,
            subset_x_names=subset_x_names,
            plot_xlims=plot_xlims,
            plot_ylims=plot_ylims,
            verbosity=verbosity,
            plot_empty=plot_empty,
            completely_random_rotation=completely_random_rotation,
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample
        )

        self.grid_len = grid_len
