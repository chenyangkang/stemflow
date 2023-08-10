#! /usr/bin/python
# %%
import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
import warnings
import pickle
import geopandas as gpd
import os

# matplotlib.style.use('ggplot')
# plt.rcParams['axes.facecolor']='w'
warnings.filterwarnings('ignore')

import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

import sys
import argparse
parser = argparse.ArgumentParser(
                    prog='Test Multiple Model',
                    description='For Classification & Hurdle tasks, test the model performace',
                    epilog='')
parser.add_argument('-sp', '--species')      # option that takes a value
parser.add_argument('-year','--year')
parser.add_argument('-size', '--sample_size')  # on/off flag
parser.add_argument('-o','--output_path', default='./Model_Performance/')
args = parser.parse_args()
print(args)

sp = args.species
year = int(args.year)
SAMPLE_SIZE = int(args.sample_size)
OUTPUT_DIR = os.path.join(args.output_path, sp)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



# %%
from BirdSTEM.utils.plot_gif import make_sample_gif, make_sample_gif_hexagon


# %% [markdown]
# # load training data
# 

# %%
checklist_data = pd.read_csv(f'./BirdSTEM/dataset/test_data/checklist_data/checklist_data_filtered_{year}.csv')

### mallard 2020
with open(f'./BirdSTEM/dataset/test_data/sp_data/{sp}/{sp}_{year}.pkl','rb') as f:
    sp_data = pickle.load(f)
    
checklist_data = checklist_data.merge(sp_data, on='sampling_event_identifier', how='left')
checklist_data['count'] = checklist_data['count'].fillna(0)


# %% [markdown]
# # Train test split

# %%
from sklearn.model_selection import train_test_split
from BirdSTEM.dataset.get_test_x_names import get_test_x_names

x_names = get_test_x_names()
X = checklist_data[['sampling_event_identifier','longitude','latitude'] + x_names]
y = checklist_data['count'].values

_, X, _, y = train_test_split(X, y, test_size=SAMPLE_SIZE, stratify=np.where(y>0, 1, 0))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3,
                                                    stratify = np.where(y>0, 1, 0), shuffle=True)




# %% [markdown]
# # Test model

# %% [markdown]
# ## Task1: Classification (modeling occurrence)

# %% [markdown]
# ### First, without AdaSTEM wrapper

# %%
from BirdSTEM.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from elapid import MaxentModel


# %%

cls_model_set_dict = {
    'LogisticRegression':LogisticRegression(random_state=42),
    'SGDClassifier':SGDClassifier(random_state=42),
    'GaussianNB':GaussianNB(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42),
    # 'SVC_linear':SVC(kernel='linear',random_state=42),
    'SVC_rbf':SVC(kernel='rbf',random_state=42),
    'RandomForestClassifier':RandomForestClassifier(random_state=42),
    'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis(),
    'MLPClassifier':MLPClassifier(random_state=42),
    'MaxEnt':MaxentModel(transform='cloglog', beta_multiplier=2.0),
    'XGBClassifier':XGBClassifier(tree_method='hist',random_state=42, n_jobs=1),
    'ComplementNB':ComplementNB(),
}


# %%
K=5

cls_metric_df_list = []
for item in list(cls_model_set_dict.keys())[::-1]:
    model_name = item
    model = cls_model_set_dict[model_name]
    
    kf = KFold(n_splits=K, shuffle=True, random_state=42).split(X, y)
    for kf_count, (train_index, test_index) in tqdm(enumerate(kf), desc=f'{model_name}', total=K):
        
        try:
            X_train = X.iloc[train_index].replace(-1,np.nan)
            imputer = SimpleImputer().fit(X_train[x_names])
            X_train[x_names] = imputer.transform(X_train[x_names])
            scaler = MinMaxScaler().fit(X_train[x_names])
            X_train[x_names] = scaler.transform(X_train[x_names])
            
            y_train = np.where(y[train_index]>0, 1, 0)
            
            X_test = X.iloc[test_index].replace(-1,np.nan)
            X_test[x_names] = imputer.transform(X_test[x_names])
            X_test[x_names] = scaler.transform(X_test[x_names])
            y_test = np.where(y[test_index]>0, 1, 0)
            
            
            sample_weights = class_weight.compute_sample_weight(class_weight='balanced',y=y_train)
            
            a = time.time()

            try:
                start_time = time.time()
                model.fit(X_train[x_names], y_train, sample_weight=sample_weights)
                finish_time = time.time()
                training_time = finish_time - start_time
            except:
                start_time = time.time()
                model.fit(X_train[x_names], y_train)
                finish_time = time.time()
                training_time = finish_time - start_time
                
            start_time = time.time()
            y_pred = model.predict(X_test[x_names])
            finish_time = time.time()
            predicting_time = finish_time - start_time
            
            y_pred = np.where(y_pred<0, 0, y_pred)
            metric_df = AdaSTEM.eval_STEM_res('classification', y_test, y_pred)
            
            metric_df['model'] = model_name
            metric_df['task_type'] = 'classification'
            metric_df['iter'] = kf_count
            metric_df['sp'] = sp
            metric_df['sample_size'] = SAMPLE_SIZE
            metric_df['training_time'] = training_time
            metric_df['predicting_time'] = predicting_time
            
            cls_metric_df_list.append(metric_df)
            
            print(metric_df,end='\n')
            
        except Exception as e:
            print(e)
            continue



    

# %%
cls_metric_df = pd.DataFrame(cls_metric_df_list)
cls_metric_df.to_csv(os.path.join(OUTPUT_DIR, f'cls_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv'),
                                        index=False)


# %% [markdown]
# ### Then, with AdaSTEM wrapper

# %%

K=5

Ada_cls_metric_df_list = []
for item in list(cls_model_set_dict.keys())[::-1]:
    model_name = item
    
    kf = KFold(n_splits=K, shuffle=True, random_state=42).split(X, y)
    for kf_count, (train_index, test_index) in tqdm(enumerate(kf), desc=f'AdaSTEM + {model_name}', total=K):
        
        try:
            X_train = X.iloc[train_index].replace(-1,np.nan)
            
            new_x_names = list(set(x_names) - set(['DOY']))
            
            imputer = SimpleImputer().fit(X_train[new_x_names])
            X_train[new_x_names] = imputer.transform(X_train[new_x_names])
            scaler = MinMaxScaler().fit(X_train[new_x_names])
            X_train[new_x_names] = scaler.transform(X_train[new_x_names])
            
            y_train = np.where(y[train_index]>0, 1, 0)
            X_test = X.iloc[test_index].replace(-1,np.nan)
            X_test[new_x_names] = imputer.transform(X_test[new_x_names])
            X_test[new_x_names] = scaler.transform(X_test[new_x_names])
            y_test = np.where(y[test_index]>0, 1, 0)
            
            model = AdaSTEMClassifier(base_model=cls_model_set_dict[model_name], 
                                        sample_weights_for_classifier=True,
                                        ensemble_fold=5,
                                        min_ensemble_require=3,
                                        grid_len_lon_upper_threshold=25,
                                        grid_len_lon_lower_threshold=5,
                                        grid_len_lat_upper_threshold=25,
                                        grid_len_lat_lower_threshold=5,
                                        points_lower_threshold=50,
                                        stixel_training_size_threshold=50,
                                        temporal_start = 1,
                                        temporal_end = 367, 
                                        temporal_step = 30.5, 
                                        temporal_bin_interval = 30.5,
                                        save_tmp=False, 
                                        save_gridding_plot=False)
            
            try:
                start_time = time.time()
                model.fit(X_train[new_x_names + ['DOY','longitude', 'latitude']], y_train)
                finish_time = time.time()
                training_time = finish_time - start_time
            except:
                start_time = time.time()
                model.set_params(**{'sample_weights_for_classifier':False})
                model.fit(X_train[new_x_names + ['DOY','longitude', 'latitude']], y_train)
                finish_time = time.time()
                training_time = finish_time - start_time
                
            start_time = time.time()
            y_pred = model.predict(X_test[new_x_names + ['DOY','longitude', 'latitude']])
            finish_time = time.time()
            predicting_time = finish_time - start_time
            
            y_pred = np.where(y_pred<0, 0, y_pred)
            metric_df = AdaSTEM.eval_STEM_res('classification', y_test, y_pred)
            
            metric_df['model'] = 'AdaSTEM_' + model_name
            metric_df['task_type'] = 'classification'
            metric_df['iter'] = kf_count
            metric_df['sp'] = sp
            metric_df['sample_size'] = SAMPLE_SIZE
            metric_df['training_time'] = training_time
            metric_df['predicting_time'] = predicting_time
        
            Ada_cls_metric_df_list.append(metric_df)
            print(metric_df,end='\n')
            
        except Exception as e:
            print(e)
            continue



    

# %%
Ada_cls_metric_df = pd.DataFrame(Ada_cls_metric_df_list)
Ada_cls_metric_df.to_csv(os.path.join(OUTPUT_DIR, f'Ada_cls_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv'),
                                        index=False)


# %%


# %% [markdown]
# ## Task2: Regression (Hurdle)

# %% [markdown]
# ### First, without AdaSTEM wrapper

# %%
from BirdSTEM.model.AdaSTEM import AdaSTEM, AdaSTEMRegressor, AdaSTEMHurdle
from BirdSTEM.model.Hurdle import Hurdle
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge, SGDRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


# %%
reg_model_set_dict = {
    'Hurdle_SGDClassifier_SGDRegressor':Hurdle(classifier=SGDClassifier(random_state=42), 
                                               regressor=SGDRegressor(random_state=42)),
    'Hurdle_Logistic_Linear':Hurdle(classifier=LogisticRegression(random_state=42), regressor=LinearRegression()),
    'Hurdle_SVC_SVR':Hurdle(classifier=SVC(kernel='rbf'), regressor=SVR(kernel='rbf')),
    'Hurdle_DecisionTreeClassifier_DecisionTreeRegressor':Hurdle(classifier=DecisionTreeClassifier(random_state=42),
                                                regressor=DecisionTreeRegressor(random_state=42)),
    'Hurdle_RandomforestClassifier_RandomforestRegressor':Hurdle(classifier=RandomForestClassifier(random_state=42),
                                                regressor=RandomForestRegressor(random_state=42)),
    'Hurdle_MLPClassifier_MLPRegressor': Hurdle(classifier=MLPClassifier(random_state=42),
                                                regressor=MLPRegressor(random_state=42)),
    'Hurdle_XGBClassifier_XGBregressor': Hurdle(classifier=XGBClassifier(tree_method='hist',n_jobs=1), 
                    regressor=XGBRegressor(tree_method='hist',n_jobs=1))
}

# %%
K=5

reg_metric_df_list = []
for item in list(reg_model_set_dict.keys())[::-1]:
    model_name = item
    model = reg_model_set_dict[model_name]
    
    kf = KFold(n_splits=K, shuffle=True, random_state=42).split(X, y)
    for kf_count, (train_index, test_index) in tqdm(enumerate(kf), desc=f'{model_name}', total=K):
        X_train = X.iloc[train_index].replace(-1,np.nan)
        imputer = SimpleImputer().fit(X_train[x_names])
        X_train[x_names] = imputer.transform(X_train[x_names])
        scaler = MinMaxScaler().fit(X_train[x_names])
        X_train[x_names] = scaler.transform(X_train[x_names])
        
        y_train = np.where(y[train_index]>0, 1, 0)
        X_test = X.iloc[test_index].replace(-1,np.nan)
        X_test[x_names] = imputer.transform(X_test[x_names])
        X_test[x_names] = scaler.transform(X_test[x_names])
        y_test = np.where(y[test_index]>0, 1, 0)
        
        sample_weights = class_weight.compute_sample_weight(class_weight='balanced',y=np.where(y_train>0,1,0))
        
        
        try:
            start_time = time.time()
            model.fit(X_train[x_names], y_train, sample_weight=sample_weights)
            finish_time = time.time()
            training_time = finish_time - start_time
        except:
            start_time = time.time()
            model.fit(X_train[x_names], y_train)
            finish_time = time.time()
            training_time = finish_time - start_time
            
        start_time = time.time()
        y_pred = model.predict(X_test[x_names])
        y_pred = np.where(y_pred<0, 0, y_pred)
        finish_time = time.time()
        predicting_time = finish_time - start_time
        
        metric_df = AdaSTEM.eval_STEM_res('hurdle', y_test, np.array(y_pred).flatten())
        metric_df['model'] = model_name
        metric_df['task_type'] = 'hurdle'
        metric_df['iter'] = kf_count
        metric_df['sp'] = sp
        metric_df['sample_size'] = SAMPLE_SIZE
        metric_df['training_time'] = training_time
        metric_df['predicting_time'] = predicting_time
        
        reg_metric_df_list.append(metric_df)
        print(metric_df,end='\n')


    

# %%
reg_metric_df = pd.DataFrame(reg_metric_df_list)
reg_metric_df.to_csv(os.path.join(OUTPUT_DIR, f'hurdle_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv'),
                                        index=False)


# %%


# %% [markdown]
# ### Then, with AdaSTEM wrapper

# %%
K=5

reg_metric_df_list = []
for item in list(reg_model_set_dict.keys())[::-1]:
    model_name = item
    model = reg_model_set_dict[model_name]

    kf = KFold(n_splits=K, shuffle=True, random_state=42).split(X, y)
    for kf_count, (train_index, test_index) in tqdm(enumerate(kf), desc=f'AdaSTEM + {model_name}', total=K):
        
        try:
            X_train = X.iloc[train_index].replace(-1,np.nan)
            
            new_x_names = list(set(x_names) - set(['DOY']))

            imputer = SimpleImputer().fit(X_train[new_x_names])
            X_train[new_x_names] = imputer.transform(X_train[new_x_names])
            scaler = MinMaxScaler().fit(X_train[new_x_names])
            X_train[new_x_names] = scaler.transform(X_train[new_x_names])

            y_train = y[train_index]
            X_test = X.iloc[test_index].replace(-1,np.nan)
            X_test[new_x_names] = imputer.transform(X_test[new_x_names])
            X_test[new_x_names] = scaler.transform(X_test[new_x_names])
            y_test = y[test_index]

            model = AdaSTEMClassifier(base_model=reg_model_set_dict[model_name], 
                                            sample_weights_for_classifier=True,
                                            ensemble_fold=5,
                                            min_ensemble_require=3,
                                            grid_len_lon_upper_threshold=25,
                                            grid_len_lon_lower_threshold=5,
                                            grid_len_lat_upper_threshold=25,
                                            grid_len_lat_lower_threshold=5,
                                            points_lower_threshold=50,
                                            stixel_training_size_threshold=50,
                                            temporal_start = 1,
                                            temporal_end = 367, 
                                            temporal_step = 30.5, 
                                            temporal_bin_interval = 30.5,
                                            save_tmp=False, 
                                            save_gridding_plot=False)
            
            try:
                start_time = time.time()
                model.fit(X_train[new_x_names + ['DOY','longitude', 'latitude']], y_train)
                finish_time = time.time()
                training_time = finish_time - start_time
                
            except:
                start_time = time.time()
                model.set_params(**{'sample_weights_for_classifier':False})
                model.fit(X_train[new_x_names + ['DOY','longitude', 'latitude']], y_train)
                finish_time = time.time()
                training_time = finish_time - start_time
                
            start_time = time.time()
            y_pred = model.predict(X_test[new_x_names + ['DOY','longitude', 'latitude']])
            y_pred = np.where(y_pred<0, 0, y_pred)
            finish_time = time.time()
            predicting_time_time = finish_time - start_time
                
            metric_df = AdaSTEM.eval_STEM_res('hurdle', y_test, np.array(y_pred).flatten())
            
            metric_df['model'] = 'AdaSTEM_' + model_name
            metric_df['task_type'] = 'hurdle'
            metric_df['iter'] = kf_count
            metric_df['sp'] = sp
            metric_df['sample_size'] = SAMPLE_SIZE
            metric_df['training_time'] = training_time
            metric_df['predicting_time'] = predicting_time
            
            reg_metric_df_list.append(metric_df)
            print(metric_df,end='\n')
            
        except Exception as e:
            print(e)
            continue


    

# %%
reg_metric_df = pd.DataFrame(reg_metric_df_list)
reg_metric_df.to_csv(os.path.join(OUTPUT_DIR, f'Ada_hurdle_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv'),
                                        index=False)


# %%
all_metrics = pd.concat([
    pd.read_csv(os.path.join(OUTPUT_DIR, f'cls_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv')),
    pd.read_csv(os.path.join(OUTPUT_DIR, f'Ada_cls_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv')),
    pd.read_csv(os.path.join(OUTPUT_DIR, f'hurdle_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv')),
    pd.read_csv(os.path.join(OUTPUT_DIR, f'Ada_hurdle_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv')),
], axis=0)


# %%
all_metrics.to_csv(os.path.join(OUTPUT_DIR, f'ALL_metric_df_SIZE_{SAMPLE_SIZE}_SP_{sp}_year_{year}.csv'))



# %%



# %%


# %%


# %%



