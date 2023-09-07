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

#validation check
from pandas.core.frame import DataFrame
from numpy import ndarray
#

######
from ..utils.quadtree import QTree, get_ensemble_quadtree, generate_temporal_bins
from .dummy_model import dummy_model1
from ..utils.validation import check_random_state
######



    
class AdaSTEM(BaseEstimator):
    '''
        attributes:
        
        self.model1: after init
        self.model2: after init
        self.x_names: after init
        self.sp: after init
        self.positive_count: after init
        
        self.sample_size: after fit
        self.month1_sample_size: after fit
        ...
        self.month12_sample_size: after fit
        
        self.bias_df: after get_bias_df
        
        
        Functions:
        __init__(self,model1, model2,x_names,sp)
        fit(self,X_train, y_train)
        predict(self,X_test)
        get_bias_df(self, data)
        score(self,X_test,y_test)
        plot_error(self)
        
    '''
    def __init__(self,base_model,
                task='hurdle',
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
                plot_xlims = (-180,180),
                plot_ylims = (-90,90)                    
                ):

        # # random state
        # self.random_state = random_state
        # rng = check_random_state(random_state)
        
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
                                                               
        if (not type(temporal_bin_start_jitter) in [str, float, int]):
            raise AttributeError(f'Input temporal_bin_start_jitter should be \'random\', float or int, got {type(temporal_bin_start_jitter)}')
        if type(temporal_bin_start_jitter) == str:
            if not temporal_bin_start_jitter=='random':
                raise AttributeError(f'The input temporal_bin_start_jitter as string should only be \'random\'. Other options include float or int. Got {temporal_bin_start_jitter}')
        self.temporal_bin_start_jitter = temporal_bin_start_jitter
        
        self.stixel_training_size_threshold = points_lower_threshold
        self.save_gridding_plot = save_gridding_plot
        self.save_tmp = save_tmp
        self.save_dir = save_dir
        self.sample_weights_for_classifier = sample_weights_for_classifier
        

    def split(self, X_train):
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
#             return cores
        
            cores2 = pd.DataFrame(list(X_train.index),columns=['data_point_index'])
            cores = pd.merge(cores, cores2, 
                             left_on='checklist_index',right_on = 'data_point_index',how='right')
            
            self.grid_dict[ensemble_index] = cores.stixel.values
            
        return self.grid_dict
    
    @staticmethod
    def _monkey_patched_predict_proba(model, X_test):
        pred = model.predict(X_test)
        pred = np.array(pred).reshape(-1,1)
        return np.concatenate([np.zeros(shape=pred.shape), pred], axis=1)
    
    def model_wrapper(self, model):
        '''
        wrap a predict_proba function for those models who don't have
        '''
        if 'predict_proba' in dir(model):
            return model
        else:
            warnings.warn(f'predict_proba function not in base_model. Monkey patching one.')
                
            model.predict_proba = self._monkey_patched_predict_proba
            return model
        
        
    def fit(self, X_train, y_train):
        
        # stixel specific x_names list
        self.stixel_specific_x_names = {}
        
        # check type
        type_X_train = type(X_train)
    
        if not type_X_train == pd.core.frame.DataFrame:
            raise TypeError(f'Input X_train should be type \'pd.core.frame.DataFrame\'. Got {str(type_X_train)}')
        
        type_y_train = type(y_train)
        if not (isinstance(y_train, ndarray) or isinstance(y_train, DataFrame)):
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
        X_train_copy['true_y'] = np.array(y_train).flatten()
        grid_dict = self.split(X_train_copy)

        # define model dict
        self.model_dict = {}
        for index,line in tqdm(self.ensemble_df.iterrows(),total=len(self.ensemble_df),desc='training: '):
            name = f'{line.ensemble_index}_{line.unique_stixel_id}'
            sub_X_train = X_train_copy[X_train_copy.index.isin(line.checklist_indexes)]
            
            if len(sub_X_train)<self.stixel_training_size_threshold: ####### threshold
                continue
            
            sub_y_train = sub_X_train.iloc[:,-1]
            sub_X_train = sub_X_train[self.x_names]
            unique_sub_y_train_binary = np.unique(np.where(sub_y_train>0, 1, 0))
            
            ##### nan check
            nan_count = np.sum(np.isnan(sub_y_train)) + np.sum(np.isnan(sub_y_train))
            if nan_count>0:
                continue
            
            ##### fit
            if (not self.task == 'regression') and (len(unique_sub_y_train_binary)==1):
                self.model_dict[f'{name}_model'] = dummy_model1(float(unique_sub_y_train_binary[0]))
                continue
            else:
                # Remove the varibales that have no variation
                self.stixel_specific_x_names[name] = self.x_names.copy()
                self.stixel_specific_x_names[name] = [i for i in self.stixel_specific_x_names[name] if not i in \
                                                        list(sub_X_train.columns[sub_X_train.std(axis=0)==0])]

                # continue, if not variable left
                if len(self.stixel_specific_x_names[name])==0:
                    continue
                
                # now we are sure to fit a model
                self.model_dict[f'{name}_model'] = copy.deepcopy(self.base_model)
                    
                if (not self.task == 'regression') and self.sample_weights_for_classifier:
                    sample_weights = \
                        class_weight.compute_sample_weight(class_weight='balanced',y=np.where(sub_y_train>0,1,0))
                    
                    try:
                        self.model_dict[f'{name}_model'].fit(np.array(sub_X_train[self.stixel_specific_x_names[name]]), 
                                                            np.array(sub_y_train),
                                                            sample_weight=sample_weights)
                    except Exception as e:
                        warnings.warn(e)
                        continue
                else:
                    try:
                        self.model_dict[f'{name}_model'].fit(np.array(sub_X_train[self.stixel_specific_x_names[name]]), 
                                                            np.array(sub_y_train))
                    except Exception as e:
                        warnings.warn(e)
                        continue
            
            
    def predict_proba(self,X_test,verbosity=0, return_std=False):
        
        type_X_test = type(X_test)
        if not type_X_test == pd.core.frame.DataFrame:
            raise TypeError(f'Input X_test should be type \'pd.core.frame.DataFrame\'. Got {type_X_test}')
        
        ##### predict
        X_test_copy = X_test.copy()
        
        round_res_list = []
        ensemble_df = self.ensemble_df
        for ensemble in list(ensemble_df.ensemble_index.unique()):
            this_ensemble = ensemble_df[ensemble_df.ensemble_index==ensemble]
            this_ensemble['stixel_calibration_point_transformed_left_bound'] = \
                        [i[0] for i in this_ensemble['stixel_calibration_point(transformed)']]

            this_ensemble['stixel_calibration_point_transformed_lower_bound'] = \
                        [i[1] for i in this_ensemble['stixel_calibration_point(transformed)']]

            this_ensemble['stixel_calibration_point_transformed_right_bound'] = \
                        this_ensemble['stixel_calibration_point_transformed_left_bound'] + this_ensemble['stixel_width']

            this_ensemble['stixel_calibration_point_transformed_upper_bound'] = \
                        this_ensemble['stixel_calibration_point_transformed_lower_bound'] + this_ensemble['stixel_height']

            X_test_copy = self.transform_pred_set_to_STEM_quad(X_test_copy,this_ensemble)
            
            ##### pred each stixel
            res_list = []
                
            iter_func = this_ensemble.iterrows() if verbosity==0 else tqdm(this_ensemble.iterrows(), 
                                                                     total=len(this_ensemble), 
                                                                     desc=f'predicting ensemble {ensemble} ')
            for index,line in iter_func:
                grid_index = line['unique_stixel_id']
                sub_X_test = X_test_copy[
                    (X_test_copy.DOY>=line['DOY_start']) & (X_test_copy.DOY<=line['DOY_end']) & \
                    (X_test_copy.lon_new>=line['stixel_calibration_point_transformed_left_bound']) &\
                    (X_test_copy.lon_new<=line['stixel_calibration_point_transformed_right_bound']) &\
                    (X_test_copy.lat_new>=line['stixel_calibration_point_transformed_lower_bound']) &\
                    (X_test_copy.lat_new<=line['stixel_calibration_point_transformed_upper_bound'])
                ]
                
                if len(sub_X_test)==0:
                    continue
                    
                ##### get training data
                for i in [self.Spatio1,self.Spatio2,'sampling_event_identifier','y_true']:
                    if i in list(self.x_names):
                        del self.x_names[self.x_names.index(i)]

                sub_X_test = sub_X_test[self.x_names]

                try:
                    model = self.model_dict[f'{ensemble}_{grid_index}_model']
                    stixel_specific_x_names = self.stixel_specific_x_names[f'{ensemble}_{grid_index}']
                    
                    if self.task=='regression':
                        pred = model.predict(np.array(sub_X_test[stixel_specific_x_names]))
                    else:
                        pred = model.predict_proba(np.array(sub_X_test[stixel_specific_x_names]))[:,1]
                    
                    res = pd.DataFrame({'index':list(sub_X_test.index),
                                        'pred':pred}).set_index('index')
                    

                except Exception as e:
                    # print(e)
                    res = pd.DataFrame({'index':list(sub_X_test.index),
                                        'pred':[np.nan]*len(list(sub_X_test.index))
                                        }).set_index('index')
                    
                res_list.append(res)
                
            res_list = pd.concat(res_list, axis=0)
            res_list = res_list.reset_index(drop=False).groupby('index').mean()
            
            round_res_list.append(res_list)
        
        ####### only sites that meet the minimum ensemble requirement is remained
        res = pd.concat([df['pred'] for df in round_res_list], axis=1)
    
        res_mean = res.mean(axis=1, skipna=True)  ##### mean of all grid model that predicts this stixel
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
        # print(f'There are {nan_frac*100:0.5f}% points ({nan_count} points) fell out of predictable range.')
        
        if return_std:
            return new_res['pred_mean'].values, new_res['pred_std'].values
        else:
            return new_res['pred_mean'].values
        
        
    def predict(self,X_test, verbosity=0, return_std=False):
        return self.predict_proba(X_test, verbosity=verbosity, return_std=return_std)
    
            
    def transform_pred_set_to_STEM_quad(self,X_train,ensemble_info):

        x_array = X_train['longitude']
        y_array = X_train['latitude']
        coord = np.array([x_array, y_array]).T
        angle = float(ensemble_info.iloc[0,:]['rotation'])
        r = angle/360
        theta = r * np.pi * 2
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        coord = coord @ rotation_matrix
        calibration_point_x_jitter = \
                float(ensemble_info.iloc[0,:]['space_jitter(first rotate by zero then add this)'][0])
        calibration_point_y_jitter = \
                float(ensemble_info.iloc[0,:]['space_jitter(first rotate by zero then add this)'][1])

        long_new = (coord[:,0] + calibration_point_x_jitter).tolist()
        lat_new = (coord[:,1] + calibration_point_y_jitter).tolist()

        X_train['lon_new'] = long_new
        X_train['lat_new'] = lat_new

        return X_train
    
    @classmethod
    def eval_STEM_res(self, task, y_test, y_pred, cls_threashold=None):
        '''
        task: one of 'regression', 'classification' or 'hurdle'
        
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
        '''
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
            
            y_test_b = np.where(y_test>cls_threashold, 1, 0)
            y_pred_b = np.where(y_pred>cls_threashold, 1, 0)
            
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


    def score(self, X_test, y_test):
        y_pred, _ = self.predict(X_test)
        score_dict = AdaSTEM.eval_STEM_res(self.task, y_test, y_pred)
        self.score_dict = score_dict
        return self.score_dict
        

    
    
    
class AdaSTEMClassifier(AdaSTEM):
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
                plot_xlims = (-180,180),
                plot_ylims = (-90,90),
                ):
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
                         plot_xlims, plot_ylims
                         )
        
    def predict(self, X_test, verbosity=0, return_std=False):
        if return_std:
            mean, std = self.predict_proba(X_test, verbosity=verbosity, return_std=True)
            mean = np.where(mean<0.5, 0, mean)
            mean = np.where(mean>=0.5, 1, mean)
            return mean, std
        else:
            mean = self.predict_proba(X_test, verbosity=verbosity, return_std=False)
            mean = np.where(mean<0.5, 0, mean)
            mean = np.where(mean>=0.5, 1, mean)
            return mean
            
        
        
class AdaSTEMRegressor(AdaSTEM):
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
                plot_xlims = (-180,180),
                plot_ylims = (-90,90)
                ):
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
                         plot_xlims, plot_ylims,
                         )
        
    def predict(self, X_test, verbosity=0, return_std=False):
        if return_std:
            mean, std = self.predict_proba(X_test, verbosity=verbosity, return_std=True)
            return mean, std
        else:
            mean = self.predict_proba(X_test, verbosity=verbosity, return_std=False)
            return mean

        
        
        
        
# class AdaSTEMHurdle(AdaSTEM):
#     def __init__(self, base_model, 
#                  task='hurdle',
#                  ensemble_fold=1, 
#                  min_ensemble_required=1, 
#                  grid_len_lon_upper_threshold=25, 
#                  grid_len_lon_lower_threshold=5, 
#                  grid_len_lat_upper_threshold=25, 
#                  grid_len_lat_lower_threshold=5, 
#                  points_lower_threshold=50, 
#                  temporal_start=1, 
#                  temporal_end=366, 
#                  temporal_step=20, 
#                  temporal_bin_interval=50, 
#                  temporal_bin_start_jitter = 'random',
#                  save_gridding_plot=False, 
#                  save_tmp=False, 
#                  save_dir='./', 
#                  sample_weights_for_classifier=True):
#         super().__init__(base_model,
#                          task,
#                          ensemble_fold, 
#                          min_ensemble_required,
#                          grid_len_lon_upper_threshold, 
#                          grid_len_lon_lower_threshold, 
#                          grid_len_lat_upper_threshold, grid_len_lat_lower_threshold, 
#                          points_lower_threshold, temporal_start, 
#                          temporal_end, temporal_step, temporal_bin_interval, 
#                          temporal_bin_start_jitter,
#                          save_gridding_plot, save_tmp, save_dir, sample_weights_for_classifier)
        
#         # self.task='hurdle'
#         # warnings.warn('You have choose HURDLE task. The goal is to first conduct classification, and then apply regression on points with *positive values*')