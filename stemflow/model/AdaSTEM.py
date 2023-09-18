import sys
import pandas as pd
import numpy as np
import numpy
import math
import os
import warnings
import pickle
import time
from tqdm import tqdm
import random
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from warnings import simplefilter
from matplotlib import cm
import copy

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_tweedie_deviance,\
        mean_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score, recall_score
from scipy.stats import spearmanr
from sklearn.utils import class_weight
from sklearn.inspection import partial_dependence

#
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import repeat

#validation check
from pandas.core.frame import DataFrame
from numpy import ndarray
from typing import Union, Tuple
#

######
from ..utils.quadtree import get_ensemble_quadtree
from .dummy_model import dummy_model1
from ..utils.validation import check_random_state
from .static_func_AdaSTEM import (
    _monkey_patched_predict_proba,
    train_one_stixel,
    assign_points_to_one_ensemble,
    transform_pred_set_to_STEM_quad,
    # predict_one_ensemble
    get_model_and_stixel_specific_x_names,
    predict_one_stixel,
    
)
######



    
class AdaSTEM(BaseEstimator):
    """A AdaSTEM model class inherited by AdaSTEMClassifier and AdaSTEMRegressor"""
    def __init__(self,
                base_model: BaseEstimator,
                task: str='hurdle',
                ensemble_fold: int=10,
                min_ensemble_required: int = 7,
                grid_len_lon_upper_threshold: Union[float, int]=25,
                grid_len_lon_lower_threshold: Union[float, int]=5,
                grid_len_lat_upper_threshold: Union[float, int]=25,
                grid_len_lat_lower_threshold: Union[float, int]=5,
                points_lower_threshold: int=50,
                temporal_start: Union[float, int] = 1, 
                temporal_end: Union[float, int]=366,
                temporal_step: Union[float, int]=20,
                temporal_bin_interval: Union[float, int] = 50,
                temporal_bin_start_jitter: Union[float, int, str] = 'random',
                spatio_bin_jitter_maginitude: Union[float, int] = 10,
                save_gridding_plot: bool=True,
                save_tmp: bool = False,
                save_dir: str='./',
                sample_weights_for_classifier: bool=True,
                Spatio1: str='longitude', 
                Spatio2: str = 'latitude', 
                Temporal1: str = 'DOY',
                use_temporal_to_train: bool=True,
                njobs: int=1,          
                plot_xlims: Tuple[Union[float, int], Union[float, int]] = (-180,180),
                plot_ylims: Tuple[Union[float, int], Union[float, int]] = (-90,90)                   
                ):
        """Make a AdaSTEM object

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
            grid_len_lon_upper_threshold: 
                force divide if grid longitude larger than the threshold. Defaults to 25.
            grid_len_lon_lower_threshold: 
                stop divide if grid longitude **will** be below than the threshold. Defaults to 5.
            grid_len_lat_upper_threshold: 
                force divide if grid latitude larger than the threshold. Defaults to 25.
            grid_len_lat_lower_threshold: 
                stop divide if grid latitude **will** be below than the threshold. Defaults to 5.
            points_lower_threshold:
                Do not train the model if the available data records for this stixel is less than this threshold,
                and directly set the value to np.nan. Defaults to 25.
            temporal_start: 
                start of the temporal sequence. Defualts to 1.
            temporal_end: 
                end of the temporal sequence. Defualts to 366.
            temporal_step: 
                step of the sliding window. Defualts to 20.
            temporal_bin_interval: 
                size of the sliding window. Defualts to 50.
            temporal_bin_start_jitter: 
                jitter of the start of the sliding window. 
                If 'random', a random jitter of range (-bin_interval, 0) will be generated
                for the start. Defualts to 'random'.
            spatio_bin_jitter_maginitude:
                jitter of the spatial gridding. Defualts to 10.
            save_gridding_plot:
                Whether ot save gridding plots. Defualts to True.
            save_tmp: 
                Whether to save the ensemble dataframe. Defualts to False.
            save_dir:
                If save_tmp==True, save the ensemble dataframe to this path. Defualts to './'.
            sample_weights_for_classifier:
                Whether to balance the sample weights of classifier for impalanced datasets. Defaults to True.
            Spatio1:
                Spatial column name 1 in data. Defaults to 'longitude'.
            Spatio2:
                Spatial column name 2 in data. Defaults to 'latitude'.
            Temporal1:
                Temporal column name 1 in data.  Defaults to 'DOY'.
            use_temporal_to_train:
                Whether to use temporal varibale to train. For example in modeling the daily aboundance of bird populaiton,
                whether use 'day of year (DOY)' as a training variable. Defaults to True.
            njobs:
                Number of multiprocessing in fitting the model. Defaults to 1.
            plot_xlims:
                If save_gridding_plot=Ture, what is the xlims of the plot. Defaults to (-180,180).
            plot_ylims:
                If save_gridding_plot=Ture, what is the ylims of the plot. Defaults to (-90,90).


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
                We remove the varibales that have no variation for each stixel.
                Therefore, the x_names are different for each stixel.
            ensemble_df (pd.core.frame.DataFrame):
                A dataframe storing the stixel gridding information.
            gridding_plot (matplotlib.figure.Figure):
                Ensemble plot.
            model_dict (dict):
                Dictionary of {stixel_index: trained_model}.
            grid_dict (dict):
                An array of stixels assigned to each emsemble.
            feature_importances_ (pd.core.frame.DataFrame):
                Feature importance dataframe for each stixel.
            
        """
        # save base model
        self.base_model = base_model
        self.Spatio1 = Spatio1
        self.Spatio2 = Spatio2
        self.Temporal1 = Temporal1
        self.use_temporal_to_train = use_temporal_to_train
        
        for func in ['fit','predict']:
            if not func in dir(self.base_model):
                raise AttributeError(f'input base model must have method \'{func}\'!')
            
        self.base_model = self.model_wrapper(self.base_model)
        
        self.task = task
        if not self.task in ['regression','classification','hurdle']:
            raise AttributeError(f'task type must be one of \'regression\', \'classification\', or \'hurdle\'! Now it is {self.task}')
        if self.task=='hurdle':
            warnings.warn('You have chosen HURDLE task. The goal is to first conduct classification, and then apply regression on points with *positive values*')
                    
        self.ensemble_fold = ensemble_fold
        self.min_ensemble_required = min_ensemble_required
        self.grid_len_lon_upper_threshold=grid_len_lon_upper_threshold
        self.grid_len_lon_lower_threshold=grid_len_lon_lower_threshold
        self.grid_len_lat_upper_threshold=grid_len_lat_upper_threshold
        self.grid_len_lat_lower_threshold=grid_len_lat_lower_threshold
        self.points_lower_threshold=points_lower_threshold
        self.temporal_start = temporal_start
        self.temporal_end = temporal_end
        self.temporal_step = temporal_step
        self.temporal_bin_interval = temporal_bin_interval
        self.spatio_bin_jitter_maginitude = spatio_bin_jitter_maginitude
        self.plot_xlims = plot_xlims
        self.plot_ylims = plot_ylims
                      
        # validate temporal_bin_start_jitter                                         
        if (not type(temporal_bin_start_jitter) in [str, float, int]):
            raise AttributeError(f'Input temporal_bin_start_jitter should be \'random\', float or int, got {type(temporal_bin_start_jitter)}')
        if type(temporal_bin_start_jitter) == str:
            if not temporal_bin_start_jitter=='random':
                raise AttributeError(f'The input temporal_bin_start_jitter as string should only be \'random\'. Other options include float or int. Got {temporal_bin_start_jitter}')
        self.temporal_bin_start_jitter = temporal_bin_start_jitter
        
        #
        self.stixel_training_size_threshold = points_lower_threshold
        self.save_gridding_plot = save_gridding_plot
        self.save_tmp = save_tmp
        self.save_dir = save_dir
        
        # validate njobs setting
        if not isinstance(njobs, int):
            raise TypeError(f'njobs is not a interger. Got {njobs}.')
        
        my_cpu_count = cpu_count()
        if njobs > my_cpu_count:
            raise ValueError(f'Setting of njobs ({njobs}) exceed the maxmimum ({my_cpu_count}).')
        
        self.njobs = njobs
        
        #
        self.sample_weights_for_classifier = sample_weights_for_classifier
        

    def split(self, 
              X_train: pd.core.frame.DataFrame, verbosity: int=1) -> dict:
        """QuadTree indexing the input data

        Args:
            X_train: Input training data
        
        Returns:
            self.grid_dict, a dictionary of one DataFrame for each grid, containing the gridding information
        """
        fold = self.ensemble_fold
        save_path = os.path.join(self.save_dir, 'ensemble_quadtree_df.csv')  if self.save_tmp else ''
        self.ensemble_df, self.gridding_plot = get_ensemble_quadtree(X_train,\
                                            Spatio1 = self.Spatio1,
                                            Spatio2 = self.Spatio2,
                                            Temporal1 = self.Temporal1,
                                            size=fold,
                                            grid_len_lon_upper_threshold=self.grid_len_lon_upper_threshold,
                                            grid_len_lon_lower_threshold=self.grid_len_lon_lower_threshold,
                                            grid_len_lat_upper_threshold=self.grid_len_lat_upper_threshold,
                                            grid_len_lat_lower_threshold=self.grid_len_lat_lower_threshold,
                                            points_lower_threshold=self.points_lower_threshold,
                                            temporal_start = self.temporal_start, 
                                            temporal_end=self.temporal_end, 
                                            temporal_step=self.temporal_step, 
                                            temporal_bin_interval = self.temporal_bin_interval,
                                            temporal_bin_start_jitter = self.temporal_bin_start_jitter,
                                            spatio_bin_jitter_maginitude = self.spatio_bin_jitter_maginitude,
                                            save_gridding_plot=self.save_gridding_plot,
                                            njobs=self.njobs,
                                            verbosity=verbosity,
                                            plot_xlims = self.plot_xlims,
                                            plot_ylims = self.plot_ylims,
                                            save_path=save_path)

        self.grid_dict = {}
        for ensemble_index in self.ensemble_df.ensemble_index.unique():
            this_ensemble = self.ensemble_df[self.ensemble_df.ensemble_index==ensemble_index]
            
            this_ensemble_gird_info = {}
            this_ensemble_gird_info['checklist_index'] = []
            this_ensemble_gird_info['stixel'] = []
            for index,line in this_ensemble.iterrows():
                this_ensemble_gird_info['checklist_index'].extend(line['checklist_indexes'])
                this_ensemble_gird_info['stixel'].extend([line['unique_stixel_id']]*len(line['checklist_indexes']))
            
            cores = pd.DataFrame(this_ensemble_gird_info)
            cores2 = pd.DataFrame(list(X_train.index),columns=['data_point_index'])
            cores = pd.merge(cores, cores2, 
                             left_on='checklist_index',right_on = 'data_point_index',how='right')
            
            self.grid_dict[ensemble_index] = cores.stixel.values
            
        return self.grid_dict

    def model_wrapper(self, 
                      model: BaseEstimator) -> BaseEstimator:
        '''wrap a predict_proba function for those models who don't have
        
        Args:
            model:
                Input model
        
        Returns:
            Wrapped model that has a `predict_proba` method
            
        '''
        if 'predict_proba' in dir(model):
            return model
        else:
            warnings.warn(f'predict_proba function not in base_model. Monkey patching one.')
                
            model.predict_proba = _monkey_patched_predict_proba
            return model
        
    def fit(self, 
            X_train: pd.core.frame.DataFrame, 
            y_train: Union[pd.core.frame.DataFrame, np.ndarray],
            verbosity: int=1):
        """Fitting method

        Args:
            X_train: Training variables
            y_train: Training target

        Raises:
            TypeError: X_train is not a type of pd.core.frame.DataFrame
            TypeError: y_train is not a type of np.ndarray or pd.core.frame.DataFrame
        """
        # check type
        type_X_train = type(X_train)
    
        if not type_X_train == pd.core.frame.DataFrame:
            raise TypeError(f'Input X_train should be type \'pd.core.frame.DataFrame\'. Got {str(type_X_train)}')
        
        type_y_train = type(y_train)
        if not (isinstance(y_train, np.ndarray) or isinstance(y_train, pd.core.frame.DataFrame)):
            raise TypeError(f'Input y_train should be type \'pd.core.frame.DataFrame\' or \'np.ndarray\'. Got {str(type_y_train)}')
        
        # store x_names
        self.x_names = list(X_train.columns)
        for i in [self.Spatio1, self.Spatio2]:
            if i in list(self.x_names):
                del self.x_names[self.x_names.index(i)]
        if not self.use_temporal_to_train:
            if self.Temporal1 in list(self.x_names):
                del self.x_names[self.x_names.index(self.Temporal1)]
            
        # quadtree
        X_train_copy = X_train.copy().reset_index(drop=True) ### I reset index here!! caution!
        del X_train
        X_train_copy['true_y'] = np.array(y_train).flatten()
        grid_dict = self.split(X_train_copy, verbosity=verbosity)

        # define model dict
        self.model_dict = {}
        # stixel specific x_names list
        self.stixel_specific_x_names = {}
        
        # Training function for each stixel
        if not self.njobs > 1:
            # single processing
            func_ = tqdm(self.ensemble_df.iterrows(),total=len(self.ensemble_df),desc='training: ') \
                        if verbosity>0 else self.ensemble_df.iterrows()
            
            for index,line in func_:
                ensemble_index = line['ensemble_index']
                unique_stixel_id = line['unique_stixel_id']
                name = f'{ensemble_index}_{unique_stixel_id}'
                checklist_indexes = line['checklist_indexes']
                model, stixel_specific_x_names = train_one_stixel(stixel_training_size_threshold = self.stixel_training_size_threshold, 
                                                                  x_names = self.x_names, 
                                                                  task = self.task,
                                                                  base_model = self.base_model, 
                                                                  sample_weights_for_classifier = self.sample_weights_for_classifier,
                                                                  X_train_copy = X_train_copy, 
                                                                  checklist_indexes = checklist_indexes)

                if model is None:
                    continue
                else:
                    self.model_dict[f'{name}_model'] = model
                    
                if len(stixel_specific_x_names)==0:
                    continue
                else:
                    self.stixel_specific_x_names[name] = stixel_specific_x_names
                    
        else:
            # multi-processing
            ensemble_index_list = self.ensemble_df['ensemble_index'].values
            unique_stixel_id_list = self.ensemble_df['unique_stixel_id'].values
            name_list = [f'{ensemble_index}_{unique_stixel_id}' for ensemble_index, unique_stixel_id in zip(
                ensemble_index_list, unique_stixel_id_list
            )]
            checklist_indexes = self.ensemble_df['checklist_indexes']
            
            with Pool(self.njobs) as p:
                plain_args_iterator = zip(
                        repeat(self.stixel_training_size_threshold), 
                        repeat(self.x_names), 
                        repeat(self.task), 
                        repeat(self.base_model),
                        repeat(self.sample_weights_for_classifier), 
                        repeat(X_train_copy),
                        checklist_indexes
                    )
                if verbosity>0:
                    args_iterator = tqdm(plain_args_iterator, total=len(checklist_indexes))
                else:
                    args_iterator = plain_args_iterator
                    
                tmp_res = p.starmap(train_one_stixel, args_iterator)
                
                # Store model and stixel specific x_names
                for res, name in zip(tmp_res, name_list):
                    model_ = res[0]
                    stixel_specific_x_names_ = res[1]
                    
                    if model_ is None:
                        continue
                    else:
                        self.model_dict[f'{name}_model'] = model_
                        
                    if len(stixel_specific_x_names_) == 0:
                        continue
                    else:
                        self.stixel_specific_x_names[name] = stixel_specific_x_names_
                          
        
        
    def predict_proba(self,
                      X_test: pd.core.frame.DataFrame,
                      verbosity: int=0, 
                      return_std: bool=False,
                      njobs: Union[None, int]=1,
                      aggregation: str='mean',
                      return_by_seperate_ensembles: bool=False) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Predict probability

        Args:
            X_test (pd.core.frame.DataFrame): 
                Testing variables.
            verbosity (int, optional): 
                show progress bar or not. Yes for 0, and No for other. Defaults to 0.
            return_std (bool, optional): 
                Whether return the standard deviation among ensembles. Defaults to False.
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.
                I do not recommend setting value larger than 1. 
                In practice, multi-processing seems to slow down the process instead of speeding up.
                Could be more practical with large amount of data.
                Still in experiment.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_seperate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by seperate ensembles.
                
        Raises:
            TypeError: 
                X_test is not of type pd.core.frame.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].
            
        Returns:
            predicted results. (pred_mean, pred_std) if return_std==Ture, and pred_mean if return_std==False.
            
            If return_by_seperate_ensembles == True:
                Return numpy.ndarray of shape (n_samples, n_ensembles)
            
        """
        type_X_test = type(X_test)
        if not type_X_test == pd.core.frame.DataFrame:
            raise TypeError(f'Input X_test should be type \'pd.core.frame.DataFrame\'. Got {type_X_test}')
        #
        if not aggregation in ['mean','median']:
            raise ValueError(f'aggregation must be one of \'mean\' and \'median\'. Got {aggregation}')
        
        if not isinstance(return_by_seperate_ensembles, bool):
            type_return_by_seperate_ensembles = str(type(return_by_seperate_ensembles))
            raise TypeError(f'return_by_seperate_ensembles must be bool. Got {type_return_by_seperate_ensembles}')
        else:
            if return_by_seperate_ensembles and return_std:
                warnings(f'return_by_seperate_ensembles == True. Autometically setting return_std=False')
                return_std = False
                
        ##### predict
        X_test_copy = X_test.copy()
        
        round_res_list = []
        
        for ensemble in list(self.ensemble_df.ensemble_index.unique()):
            this_ensemble = self.ensemble_df[self.ensemble_df.ensemble_index==ensemble]
            this_ensemble['stixel_calibration_point_transformed_left_bound'] = \
                        [i[0] for i in this_ensemble['stixel_calibration_point(transformed)']]

            this_ensemble['stixel_calibration_point_transformed_lower_bound'] = \
                        [i[1] for i in this_ensemble['stixel_calibration_point(transformed)']]

            this_ensemble['stixel_calibration_point_transformed_right_bound'] = \
                        this_ensemble['stixel_calibration_point_transformed_left_bound'] + this_ensemble['stixel_width']

            this_ensemble['stixel_calibration_point_transformed_upper_bound'] = \
                        this_ensemble['stixel_calibration_point_transformed_lower_bound'] + this_ensemble['stixel_height']

            X_test_copy = transform_pred_set_to_STEM_quad(self.Spatio1, self.Spatio2, X_test_copy, this_ensemble)
            
            ##### pred each stixel
            if not njobs > 1:
                # single process
                res_list = []
                iter_func = this_ensemble.iterrows() if verbosity==0 else tqdm(this_ensemble.iterrows(), 
                                                            total=len(this_ensemble), 
                                                            desc=f'predicting ensemble {ensemble} ')
                for index,stixel in iter_func:
                    model_x_names_tuple = get_model_and_stixel_specific_x_names(
                                                                                self.model_dict, 
                                                                                ensemble, 
                                                                                stixel['unique_stixel_id'], 
                                                                                self.stixel_specific_x_names, 
                                                                                self.x_names
                                                                                )
                    
                    if model_x_names_tuple[0] is None:
                        continue

                    res = predict_one_stixel(
                                    X_test_copy, 
                                    self.Temporal1,
                                    self.Spatio1,
                                    self.Spatio2,
                                    stixel[f'{self.Temporal1}_start'],
                                    stixel[f'{self.Temporal1}_end'],
                                    stixel['stixel_calibration_point_transformed_left_bound'],
                                    stixel['stixel_calibration_point_transformed_right_bound'],
                                    stixel['stixel_calibration_point_transformed_lower_bound'],
                                    stixel['stixel_calibration_point_transformed_upper_bound'],
                                    self.x_names,
                                    self.task,
                                    model_x_names_tuple
                                )
                    
                    if res is None:
                        continue
                    
                    res_list.append(res)
            else:
                # multi-processing
                with Pool(njobs) as p:
                    plain_args_iterator = zip(
                                    repeat(X_test_copy), 
                                    repeat(self.Temporal1),
                                    repeat(self.Spatio1),
                                    repeat(self.Spatio2),
                                    this_ensemble[f'{self.Temporal1}_start'],
                                    this_ensemble[f'{self.Temporal1}_end'],
                                    this_ensemble['stixel_calibration_point_transformed_left_bound'],
                                    this_ensemble['stixel_calibration_point_transformed_right_bound'],
                                    this_ensemble['stixel_calibration_point_transformed_lower_bound'],
                                    this_ensemble['stixel_calibration_point_transformed_upper_bound'],
                                    repeat(self.x_names),
                                    repeat(self.task),
                                    [get_model_and_stixel_specific_x_names(
                                          self.model_dict, 
                                          ensemble, 
                                          grid_index, 
                                          self.stixel_specific_x_names, 
                                          self.x_names) for grid_index in this_ensemble['unique_stixel_id']]
                        )
                    if verbosity>0:
                        args_iterator = tqdm(plain_args_iterator, total=len(this_ensemble), desc=f'predicting ensemble {ensemble} ')
                    else:
                        args_iterator = plain_args_iterator
                        
                    res_list = p.starmap(predict_one_stixel, args_iterator)
            
            try:        
                res_list = pd.concat(res_list, axis=0)
            except:
                res_list = pd.DataFrame({
                    'index':list(X_test.index),
                    'pred':[np.nan] * len(X_test.index)
                }).set_index('index')
                
            res_list = res_list.reset_index(drop=False).groupby('index').mean()
            round_res_list.append(res_list)
        
        ####### only sites that meet the minimum ensemble requirement are kept
        res = pd.concat([df['pred'] for df in round_res_list], axis=1)

        # Experimental Function
        if return_by_seperate_ensembles:
            new_res = pd.DataFrame({
                'index':list(X_test.index)
            }).set_index('index')
            new_res = new_res.merge(res, left_on='index', right_on='index', how='left')
            return new_res.values
        
        if aggregation=='mean':
            res_mean = res.mean(axis=1, skipna=True)  ##### mean of all grid model that predicts this stixel
        elif aggregation=='median':
            res_mean = res.median(axis=1, skipna=True)
            
        res_std = res.std(axis=1, skipna=True)
        
        res_nan_count = res.isnull().sum(axis=1)
        res_not_nan_count = len(round_res_list) - res_nan_count
        
        pred_mean = np.where(res_not_nan_count.values < self.min_ensemble_required,
                                    np.nan,
                                    res_mean.values)
        pred_std = np.where(res_not_nan_count.values < self.min_ensemble_required,
                                    np.nan,
                                    res_std.values)
        
        res = pd.DataFrame({
            'index':list(res_mean.index),
            'pred_mean':pred_mean,
            'pred_std':pred_std
        }).set_index('index')
        
        new_res = pd.DataFrame({
            'index':list(X_test.index)
        }).set_index('index')

        new_res = new_res.merge(res, left_on='index', right_on='index', how='left')
        
        nan_count = np.sum(np.isnan(new_res['pred_mean'].values))
        nan_frac = nan_count / len(new_res['pred_mean'].values)
        warnings.warn(f'There are {nan_frac}% points ({nan_count} points) fell out of predictable range.')
        
        if return_std:
            return new_res['pred_mean'].values, new_res['pred_std'].values
        else:
            return new_res['pred_mean'].values
        
        
        
    def predict(self,
                X_test: pd.core.frame.DataFrame,
                verbosity: int=0, 
                return_std: bool=False,
                njobs: Union[None, int]=1,
                aggregation: str='mean',
                return_by_seperate_ensembles: bool=False) -> Union[np.ndarray, Tuple[np.ndarray]]:
                      
        """A rewrite of predict_proba

        Args:
            X_test (pd.core.frame.DataFrame): 
                Testing variables.
            verbosity (int, optional): 
                show progress bar or not. Yes for 0, and No for other. Defaults to 0.
            return_std (bool, optional): 
                Whether return the standard deviation among ensembles. Defaults to False.
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.
                I do not recommend setting value larger than 1. 
                In practice, multi-processing seems to slow down the process instead of speeding up.
                Could be more practical with large amount of data.
                Still in experiment.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_seperate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by seperate ensembles.
                
        Raises:
            TypeError: 
                X_test is not of type pd.core.frame.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].
            
        Returns:
            predicted results. (pred_mean, pred_std) if return_std==Ture, and pred_mean if return_std==False.
            
            If return_by_seperate_ensembles == True:
                Return numpy.ndarray of shape (n_samples, n_ensembles)
                
        """
        
        return self.predict_proba(X_test, verbosity=verbosity, return_std=return_std, njobs=njobs, aggregation=aggregation, return_by_seperate_ensembles=return_by_seperate_ensembles)
    
    
    @classmethod
    def eval_STEM_res(self,
                      task: str,
                      y_test: Union[pd.core.series.Series, np.ndarray], 
                      y_pred: Union[pd.core.series.Series, np.ndarray], 
                      cls_threashold: Union[float, None]=None) -> dict:
        """Evaluation using multiple metrics
        
        Classification metrics used: 
        1. AUC
        2. Cohen's Kappa
        3. F1
        4. precision
        5. recall
        6. average precision 
        
        Regression metrics used: 
        1. spearman's r
        2. peason's r
        3. R2
        4. mean absolute error (MAE)
        5. mean squared error (MSE)
        6. poisson deviance explained (PDE)
        
        Args:
            task (str): 
                one of 'regression', 'classification' or 'hurdle'.
            y_test (Union[pd.core.series.Series, np.ndarray]): 
                y true
            y_pred (Union[pd.core.series.Series, np.ndarray]): 
                y predicted
            cls_threashold (Union[float, None], optional): 
                Cutting threashold for the classification. 
                Values above cls_threashold will be labeled as 1 and 0 otherwise. 
                Defaults to None (0.5 for classification and 0 for hurdle).

        Raises:
            AttributeError: task not one of 'regression', 'classification' or 'hurdle'.

        Returns:
            dict: dictionary containing the metric names and their values.
        """

        if not task in ['regression','classification','hurdle']:
            raise AttributeError(f'task type must be one of \'regression\', \'classification\', or \'hurdle\'! Now it is {task}')
    
        if cls_threashold==None:
            if task=='classification':
                cls_threashold = 0.5
            elif task=='hurdle':
                cls_threashold = 0
        
        from sklearn.metrics import roc_auc_score, cohen_kappa_score, r2_score, d2_tweedie_score, \
            f1_score, precision_score, recall_score, average_precision_score, mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr, spearmanr

        if not task=='regression':
            
            a = pd.DataFrame({
                'y_ture':y_test,
                'pred':y_pred
            }).dropna()
            
            y_test_b = np.where(a.y_ture>cls_threashold, 1, 0)
            y_pred_b = np.where(a.pred>cls_threashold, 1, 0)
            
            if len(np.unique(y_test_b))==1 and len(np.unique(y_pred_b))==1:
                auc, kappa, f1, precision, recall, average_precision = [np.nan] * 6
            
            else:
                auc = roc_auc_score(y_test_b, y_pred_b)
                kappa = cohen_kappa_score(y_test_b, y_pred_b)
                f1 = f1_score(y_test_b, y_pred_b)
                precision = precision_score(y_test_b, y_pred_b)
                recall = recall_score(y_test_b, y_pred_b)
                average_precision = average_precision_score(y_test_b, y_pred_b)
            
        else:
            auc, kappa, f1, precision, recall, average_precision = [np.nan] * 6
            
        if not task=='classification':
            a = pd.DataFrame({
                'y_ture':y_test,
                'pred':y_pred
            }).dropna()
            s_r, _ = spearmanr(np.array(a.y_ture), np.array(a.pred))
            p_r, _ = pearsonr(np.array(a.y_ture), np.array(a.pred))
            r2 = r2_score(a.y_ture, a.pred)
            MAE = mean_absolute_error(a.y_ture, a.pred)
            MSE = mean_squared_error(a.y_ture, a.pred)
            try:
                poisson_deviance_explained = d2_tweedie_score(a[a.pred>0].y_ture, a[a.pred>0].pred, power=1)
            except:
                poisson_deviance_explained = np.nan
        else:
            s_r, p_r, r2, MAE, MSE, poisson_deviance_explained = [np.nan] * 6
        
        return {
            'AUC':auc,
            'kappa':kappa,
            'f1':f1,
            'precision':precision,
            'recall':recall,
            'average_precision':average_precision,
            'Spearman_r':s_r,
            'Pearson_r':p_r,
            'R2':r2,
            'MAE':MAE,
            'MSE':MSE,
            'poisson_deviance_explained':poisson_deviance_explained
        }


    def score(self, 
              X_test: pd.core.frame.DataFrame, 
              y_test: Union[pd.core.series.Series, np.ndarray]) -> dict:
        """Combine predicting and evaluating in one method

        Args:
            X_test (pd.core.frame.DataFrame): Testing variables
            y_test (Union[pd.core.series.Series, np.ndarray]): y true

        Returns:
            dict: dictionary containing the metric names and their values.
        """
        
        y_pred, _ = self.predict(X_test)
        score_dict = AdaSTEM.eval_STEM_res(self.task, y_test, y_pred)
        self.score_dict = score_dict
        return self.score_dict
        
    def calculate_feature_importances(self):
        """A method to generate feature importance values for each stixel.
        
        Feature importances are saved in self.feature_importances_.
        
        Attribute dependence:
            1. self.ensemble_df
            2. self.model_dict
            3. self.stixel_specific_x_names
            4. The input base model should have attribute `feature_importances_`
            
        """
        # generate feature importance dict
        feature_importance_list = []
        
        for index,ensemble_row in self.ensemble_df.drop('checklist_indexes', axis=1).iterrows():
            if ensemble_row['stixel_checklist_count']<self.stixel_training_size_threshold:
                continue
            try:
                ensemble_index = ensemble_row['ensemble_index']
                stixel_index = ensemble_row['unique_stixel_id']
                the_model = self.model_dict[f'{ensemble_index}_{stixel_index}_model']
                x_names = self.stixel_specific_x_names[f'{ensemble_index}_{stixel_index}']
                
                if isinstance(the_model, dummy_model1):
                    importance_dict = dict(zip(self.x_names, [1/len(self.x_names)] * len(self.x_names)))
                else:
                    feature_imp = the_model.feature_importances_
                    importance_dict = dict(zip(x_names, feature_imp))
                    
                importance_dict['stixel_index'] = stixel_index
                feature_importance_list.append(importance_dict)
                
            except Exception as e:
                continue
        
        self.feature_importances_ = pd.DataFrame(feature_importance_list).set_index('stixel_index').reset_index(drop=False).fillna(0)
        
        
    def assign_feature_importances_by_points(self,
                                             Sample_ST_df: Union[pd.core.frame.DataFrame, None] = None,
                                             verbosity: int=0,
                                             aggregation: str='mean',
                                             njobs: Union[int, None]=1,
                                             ) -> pd.core.frame.DataFrame:
        """Assign feature importance to the input spatio-temporal points

        Args:
            Sample_ST_df (Union[pd.core.frame.DataFrame, None], optional): 
                Dataframe that indicate the spatio-temporal points of interest. 
                Must contain `self.Spatio1`, `self.Spatio2`, and `self.Temporal1` in columns. 
                If None, the resolution will be:

                | varibale|values|
                |---------|--------|
                |Spatio_var1|np.arange(-180,180,1)|
                |Spatio_var2|np.arange(-90,90,1)|
                |Temporal_var1|np.arange(1,366,7)|

                Defaults to None.
            verbosity (int, optional):
                Whether to show progressbar during assigning. 0 for No, otherwise Yes. Defaults to 0.
            aggregation (str, optional):
                One of 'mean' and 'median' to aggregate feature importance across ensembles.
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.
        
        Raises:
            NameError: 
                feature_importances_ attribute is not calculated. Try model.calculate_feature_importances() first.
            ValueError:
                f'aggregation not one of [\'mean\',\'median\'].'
            KeyError: 
                One of [`self.Spatio1`, `self.Spatio2`, `self.Temporal1`] not found in `Sample_ST_df.columns`

        Returns:
            DataFrame with feature importance assigned.
        """
        #
        if not 'feature_importances_' in dir(self):
            raise NameError(f'feature_importances_ attribute is not calculated. Try model.calculate_feature_importances() first.')
        #
        if not aggregation in ['mean','median']:
            raise ValueError(f'aggregation not one of [\'mean\',\'median\'].')
        #
        if njobs is None:
            njobs = self.njobs
        
        #
        if not (Sample_ST_df is None):
            for var_name in [self.Spatio1, self.Spatio2, self.Temporal1]:
                if not var_name in Sample_ST_df.columns:
                    raise KeyError(f'{var_name} not found in Sample_ST_df.columns')
        else:
            Spatio_var1 = np.arange(-180,180,1)
            Spatio_var2 = np.arange(-90,90,1)
            Temporal_var1 = np.arange(1,366,7)
            new_Spatio_var1, new_Spatio_var2, new_Temporal_var1 = np.meshgrid(
                Spatio_var1,Spatio_var2,Temporal_var1
            )
            
            Sample_ST_df = pd.DataFrame({
                self.Temporal1: new_Temporal_var1.flatten(),
                self.Spatio1: new_Spatio_var1.flatten(),
                self.Spatio2: new_Spatio_var2.flatten()
            })
        
        # assign input spatio-temporal points to stixels
        
        if not njobs > 1:
            # Single processing
            round_res_list = []
            iter_func_ = tqdm(list(self.ensemble_df.ensemble_index.unique())) if verbosity>0 else list(self.ensemble_df.ensemble_index.unique())
            for ensemble in iter_func_:
                res_list = assign_points_to_one_ensemble(
                                    self.ensemble_df,
                                    ensemble,
                                    Sample_ST_df,
                                    self.Temporal1,
                                    self.Spatio1,
                                    self.Spatio2,
                                    self.feature_importances_
                                    )
                round_res_list.append(res_list)
            
        else:
            # multi-processing
            with Pool(njobs) as p:
                plain_args_iterator = zip(
                                    repeat(self.ensemble_df),
                                    list(self.ensemble_df.ensemble_index.unique()),
                                    repeat(Sample_ST_df),
                                    repeat(self.Temporal1),
                                    repeat(self.Spatio1),
                                    repeat(self.Spatio2),
                                    repeat(self.feature_importances_)
                    )
                if verbosity>0:
                    args_iterator = tqdm(plain_args_iterator, total=len(list(self.ensemble_df.ensemble_index.unique())))
                else:
                    args_iterator = plain_args_iterator
                    
                round_res_list = p.starmap(assign_points_to_one_ensemble, args_iterator)
        
        round_res_df = pd.concat(round_res_list, axis=0)
        ensemble_available_count = round_res_df.groupby('sample_index').count().iloc[:,0]
        
        # Only points with more than self.min_ensemble_required ensembles available are used
        usable_sample = ensemble_available_count[ensemble_available_count>=self.min_ensemble_required] #
        round_res_df = round_res_df[round_res_df['sample_index'].isin(list(usable_sample.index))]
        
        # aggregate across ensembles
        if aggregation=='mean':
            mean_feature_importances_across_ensembles = round_res_df.groupby('sample_index').mean()
        elif aggregation=='median':
            mean_feature_importances_across_ensembles = round_res_df.groupby('sample_index').median()
            
        if self.use_temporal_to_train:
            mean_feature_importances_across_ensembles = mean_feature_importances_across_ensembles.rename(columns={self.Temporal1:f'{self.Temporal1}_predictor'})
        out_ = pd.concat([Sample_ST_df, mean_feature_importances_across_ensembles], axis=1).dropna()
        return out_
    
    


    
class AdaSTEMClassifier(AdaSTEM):
    """AdaSTEM model Classifier interface"""
    def __init__(self,base_model,
                task='classification',
                ensemble_fold=10,
                min_ensemble_required = 7,
                grid_len_lon_upper_threshold=25,
                grid_len_lon_lower_threshold=5,
                grid_len_lat_upper_threshold=25,
                grid_len_lat_lower_threshold=5,
                points_lower_threshold=50,
                temporal_start = 1, 
                temporal_end=366,
                temporal_step=20, 
                temporal_bin_interval = 50,
                temporal_bin_start_jitter = 'random',
                spatio_bin_jitter_maginitude = 10,
                save_gridding_plot=False,
                save_tmp = False,
                save_dir='./',
                sample_weights_for_classifier=True,
                Spatio1='longitude',
                Spatio2 = 'latitude', 
                Temporal1 = 'DOY',
                use_temporal_to_train=True,
                njobs=1,
                plot_xlims = (-180,180),
                plot_ylims = (-90,90),
                ):
        """
        
        Example:
            ```
            >>> from stemflow.model.AdaSTEM import AdaSTEMClassifier
            >>> from xgboost import XGBClassifier
            >>> model = AdaSTEMClassifier(base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
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
                                    use_temporal_to_train=True)
            >>> model.fit(X_train, y_train)
            >>> pred = model.predict(X_test)
            ```
                                
        """
        super().__init__(base_model, 
                         task,
                         ensemble_fold, 
                         min_ensemble_required,
                         grid_len_lon_upper_threshold, 
                         grid_len_lon_lower_threshold, 
                         grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, 
                         points_lower_threshold, temporal_start, 
                         temporal_end, temporal_step, temporal_bin_interval, 
                         temporal_bin_start_jitter,
                         spatio_bin_jitter_maginitude,
                         save_gridding_plot, save_tmp, save_dir, 
                         sample_weights_for_classifier,
                         Spatio1, Spatio2, Temporal1,
                         use_temporal_to_train,
                         njobs,
                         plot_xlims, plot_ylims
                         )
        
        
    def predict(self, 
                X_test: pd.core.frame.DataFrame, 
                verbosity:int =0, 
                return_std: bool=False, 
                cls_threashold: float=0.5, 
                njobs: Union[int, None]=1,
                aggregation: str='mean',
                return_by_seperate_ensembles: bool=False) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """A rewrite of predict_proba

        Args:
            X_test (pd.core.frame.DataFrame): 
                Testing variables.
            verbosity (int, optional): 
                show progress bar or not. Yes for 0, and No for other. Defaults to 0.
            return_std (bool, optional): 
                Whether return the standard deviation among ensembles. Defaults to False.
            cls_threashold (float, optional): 
                Cutting threashold for the classification. 
                Values above cls_threashold will be labeled as 1 and 0 otherwise. 
                Defaults to 0.5.
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.
                I do not recommend setting value larger than 1. 
                In practice, multi-processing seems to slow down the process instead of speeding up.
                Could be more practical with large amount of data.
                Still in experiment.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_seperate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by seperate ensembles.

        Raises:
            TypeError:
                X_test is not of type pd.core.frame.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].
            
        Returns:
            predicted results. (pred_mean, pred_std) if return_std==Ture, and pred_mean if return_std==False.
            
        """
                    
        if return_std:
            mean, std = self.predict_proba(X_test, verbosity=verbosity, return_std=True, njobs=njobs, aggregation=aggregation, return_by_seperate_ensembles=return_by_seperate_ensembles)
            mean = np.where(mean<cls_threashold, 0, mean)
            mean = np.where(mean>=cls_threashold, 1, mean)
            return mean, std
        else:
            mean = self.predict_proba(X_test, verbosity=verbosity, return_std=False, njobs=njobs, aggregation=aggregation, return_by_seperate_ensembles=return_by_seperate_ensembles)
            mean = np.where(mean<cls_threashold, 0, mean)
            mean = np.where(mean>=cls_threashold, 1, mean)
            return mean
            
        
        
class AdaSTEMRegressor(AdaSTEM):
    """AdaSTEM model Regressor interface"""
    def __init__(self,base_model,
                task='regression',
                ensemble_fold=10,
                min_ensemble_required = 7,
                grid_len_lon_upper_threshold=25,
                grid_len_lon_lower_threshold=5,
                grid_len_lat_upper_threshold=25,
                grid_len_lat_lower_threshold=5,
                points_lower_threshold=50,
                temporal_start = 1, 
                temporal_end=366,
                temporal_step=20, 
                temporal_bin_interval = 50,
                temporal_bin_start_jitter = 'random',
                spatio_bin_jitter_maginitude = 10,
                save_gridding_plot=False,
                save_tmp = False,
                save_dir='./',
                sample_weights_for_classifier=True,
                Spatio1='longitude',
                Spatio2 = 'latitude', 
                Temporal1 = 'DOY',
                use_temporal_to_train=True,
                njobs=1,
                plot_xlims = (-180,180),
                plot_ylims = (-90,90)
                ):
        """
        
        Example:
            ```
            >>> from stemflow.model.AdaSTEM import AdaSTEMRegressor
            >>> from xgboost import XGBRegressor
            >>> model = AdaSTEMRegressor(base_model=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
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
                                    use_temporal_to_train=True)
            >>> model.fit(X_train, y_train)
            >>> pred = model.predict(X_test)
            ```
                                
        """
        super().__init__(base_model, 
                         task,
                         ensemble_fold, 
                         min_ensemble_required,
                         grid_len_lon_upper_threshold, 
                         grid_len_lon_lower_threshold, 
                         grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, 
                         points_lower_threshold, temporal_start, 
                         temporal_end, temporal_step, temporal_bin_interval, 
                         temporal_bin_start_jitter,
                         spatio_bin_jitter_maginitude,
                         save_gridding_plot, save_tmp, save_dir,
                         sample_weights_for_classifier,
                         Spatio1, Spatio2, Temporal1,
                         use_temporal_to_train,
                         njobs,
                         plot_xlims, plot_ylims,
                         )
        
       