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
        grid_len: [float, int, None] = None,
        points_lower_threshold: int = 50,
        stixel_training_size_threshold: int = None,
        temporal_start: Union[float, int] = 1,
        temporal_end: Union[float, int] = 366,
        temporal_step: Union[float, int] = 20,
        temporal_bin_interval: Union[float, int] = 50,
        temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
        spatio_bin_jitter_magnitude: Union[float, int] = "adaptive",
        save_gridding_plot: bool = True,
        save_tmp: bool = False,
        save_dir: str = "./",
        sample_weights_for_classifier: bool = True,
        Spatio1: str = "longitude",
        Spatio2: str = "latitude",
        Temporal1: str = "DOY",
        use_temporal_to_train: bool = True,
        njobs: int = 1,
        subset_x_names: bool = False,
        ensemble_models_disk_saver: bool = False,
        ensemble_models_disk_saving_dir: str = "./",
        plot_xlims: Tuple[Union[float, int], Union[float, int]] = (-180, 180),
        plot_ylims: Tuple[Union[float, int], Union[float, int]] = (-90, 90),
        verbosity: int = 0,
        plot_empty: bool = False,
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
            save_gridding_plot:
                Whether ot save gridding plots. Defaults to True.
            save_tmp:
                Whether to save the ensemble dataframe. Defaults to False.
            save_dir:
                If save_tmp==True, save the ensemble dataframe to this path. Defaults to './'.
            ensemble_models_disk_saver:
                Whether to balance the sample weights of classifier for imbalanced datasets. Defaults to True.
            Spatio1:
                Spatial column name 1 in data. Defaults to 'longitude'.
            Spatio2:
                Spatial column name 2 in data. Defaults to 'latitude'.
            Temporal1:
                Temporal column name 1 in data.  Defaults to 'DOY'.
            use_temporal_to_train:
                Whether to use temporal variable to train. For example in modeling the daily abundance of bird population,
                whether use 'day of year (DOY)' as a training variable. Defaults to True.
            njobs:
                Number of multiprocessing in fitting the model. Defaults to 1.
            subset_x_names:
                Whether to only store variables with std > 0 for each stixel. Set to False will significantly increase the training speed.
            ensemble_disk_saver:
                Whether to save each ensemble of models to dicts instead of saving them in memory.
            ensemble_models_disk_saving_dir:
                Where to save the ensemble models. Only valid if ensemble_disk_saver is True.
            plot_xlims:
                If save_gridding_plot=true, what is the xlims of the plot. Defaults to (-180,180).
            plot_ylims:
                If save_gridding_plot=true, what is the ylims of the plot. Defaults to (-90,90).
            verbosity:
                0 to output nothing and everything otherwise.
            plot_empty:
                Whether to plot the empty grid

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
            base_model,
            task,
            ensemble_fold,
            min_ensemble_required,
            None,
            None,
            points_lower_threshold,
            stixel_training_size_threshold,
            temporal_start,
            temporal_end,
            temporal_step,
            temporal_bin_interval,
            temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude,
            save_gridding_plot,
            save_tmp,
            save_dir,
            sample_weights_for_classifier,
            Spatio1,
            Spatio2,
            Temporal1,
            use_temporal_to_train,
            njobs,
            subset_x_names,
            ensemble_models_disk_saver,
            ensemble_models_disk_saving_dir,
            plot_xlims,
            plot_ylims,
            verbosity,
            plot_empty,
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
        base_model,
        task="classification",
        ensemble_fold=10,
        min_ensemble_required=7,
        grid_len=25,
        points_lower_threshold=50,
        stixel_training_size_threshold=None,
        temporal_start=1,
        temporal_end=366,
        temporal_step=20,
        temporal_bin_interval=50,
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        save_gridding_plot=False,
        save_tmp=False,
        save_dir="./",
        sample_weights_for_classifier=True,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        use_temporal_to_train=True,
        njobs=1,
        subset_x_names=False,
        ensemble_models_disk_saver=False,
        ensemble_models_disk_saving_dir="./",
        plot_xlims=(-180, 180),
        plot_ylims=(-90, 90),
        verbosity=0,
        plot_empty=False,
    ):
        super().__init__(
            base_model,
            task,
            ensemble_fold,
            min_ensemble_required,
            None,
            None,
            points_lower_threshold,
            stixel_training_size_threshold,
            temporal_start,
            temporal_end,
            temporal_step,
            temporal_bin_interval,
            temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude,
            save_gridding_plot,
            save_tmp,
            save_dir,
            sample_weights_for_classifier,
            Spatio1,
            Spatio2,
            Temporal1,
            use_temporal_to_train,
            njobs,
            subset_x_names,
            ensemble_models_disk_saver,
            ensemble_models_disk_saving_dir,
            plot_xlims,
            plot_ylims,
            verbosity,
            plot_empty,
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
        base_model,
        task="regression",
        ensemble_fold=10,
        min_ensemble_required=7,
        grid_len=25,
        points_lower_threshold=50,
        stixel_training_size_threshold=None,
        temporal_start=1,
        temporal_end=366,
        temporal_step=20,
        temporal_bin_interval=50,
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        save_gridding_plot=False,
        save_tmp=False,
        save_dir="./",
        sample_weights_for_classifier=True,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        use_temporal_to_train=True,
        njobs=1,
        subset_x_names=False,
        ensemble_models_disk_saver=False,
        ensemble_models_disk_saving_dir="./",
        plot_xlims=(-180, 180),
        plot_ylims=(-90, 90),
        verbosity=0,
        plot_empty=False,
    ):
        super().__init__(
            base_model,
            task,
            ensemble_fold,
            min_ensemble_required,
            None,
            None,
            points_lower_threshold,
            stixel_training_size_threshold,
            temporal_start,
            temporal_end,
            temporal_step,
            temporal_bin_interval,
            temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude,
            save_gridding_plot,
            save_tmp,
            save_dir,
            sample_weights_for_classifier,
            Spatio1,
            Spatio2,
            Temporal1,
            use_temporal_to_train,
            njobs,
            subset_x_names,
            ensemble_models_disk_saver,
            ensemble_models_disk_saving_dir,
            plot_xlims,
            plot_ylims,
            verbosity,
            plot_empty,
        )

        self.grid_len = grid_len
