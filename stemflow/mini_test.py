# %% [markdown]
# # A **mini** test of stemflow
# 
# Yangkang Chen<br>
# Sep 12, 2023

# %%
import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
import warnings
import pickle
import os
import h3pandas
from urllib.request import urlopen

# warnings.filterwarnings('ignore')

def run_mini_test(delet_tmp_files: bool=True, show: bool = False, ensemble_models_disk_saver=False, ensemble_models_disk_saving_dir='./', speed_up_times=1):
    """Run a mini test
    
    Processes:
        1. Request data
        2. ST_train_test_split
        3. Import stemflow modules
        4. Declare model instance
        5. Fitting model
        6. Calculating feature importances
        7. Assigning importance to points
        8. Plotting top 2 important variables
        9. Calculate the fitting errors
        10. Predicting on test set
        11. Evaluation
        12. Watermark
        13. Deleting tmp files (optional)
        
    Args:
        delet_tmp_files:
            Whether to delet files after mini test.
        show:
            Whether to show the charts in jupyter.
        ensemble_models_disk_saver:
            Whether to save ensembles in disk instead of memory.
        ensemble_models_disk_saving_dir:
            if ensemble_models_disk_saver == True, where to save the models.
        speed_up_times:
            Speed up the mini test. For example, 2 means 2 times speed up. 
    """
    #
    print('Start Running Mini-test...')
    print(f'Temporary files will be stored at: ./stemflow_mini_test/')
    if delet_tmp_files:
        print('Temporary files will be deleted.')
    else:
        print('Temporary files will *NOT* be deleted.')
    # download mini data
    if not os.path.exists('./stemflow_mini_test'):
        os.makedirs('./stemflow_mini_test')
    if not 'mini_data.pkl' in os.listdir('./stemflow_mini_test'):
        url = "https://chenyangkang.github.io/stemflow/mini_data/mini_data.pkl"
        print(f'Requesting data from {url} ...')
        data = pickle.load(urlopen(url))
        # data = pd.read_csv(url)
        data.to_csv('./stemflow_mini_test/mini_data.csv', index=False)
        print('Done.')
    else:
        print('Mini-data already downloaded.')


    # %%
    # load data
    data = pd.read_csv('./stemflow_mini_test/mini_data.csv')

    # %%
    # data.head()

    # %%
    plt.scatter(
        data.longitude,
        data.latitude,
        s=0.2
    )
    plt.savefig('./stemflow_mini_test/data_plot.pdf')
    if show:
        plt.show()
    else:
        plt.close()

    # %% [markdown]
    # # Get X and y

    # %%
    
    x_names = [ 'duration_minutes',
                'Traveling',
                'DOY',
                'time_observation_started_minute_of_day',
                'elevation_mean',
                'slope_mean',
                'eastness_mean',
                'northness_mean',
                'bio1',
                'bio2',
                'bio3',
                'bio4',
                'bio5',
                'bio6',
                'bio7',
                'bio8',
                'bio9',
                'bio10',
                'bio11',
                'bio12',
                'bio13',
                'bio14',
                'bio15',
                'bio16',
                'bio17',
                'bio18',
                'bio19',
                'closed_shrublands',
                'cropland_or_natural_vegetation_mosaics',
                'croplands',
                'deciduous_broadleaf_forests',
                'deciduous_needleleaf_forests',
                'evergreen_broadleaf_forests',
                'evergreen_needleleaf_forests',
                'grasslands',
                'mixed_forests',
                'non_vegetated_lands',
                'open_shrublands',
                'permanent_wetlands',
                'savannas',
                'urban_and_built_up_lands',
                'water_bodies',
                'woody_savannas',
                'entropy']   
     
    X = data.drop('count', axis=1)[x_names + ['longitude','latitude']]
    y = data['count'].values


    # %% [markdown]
    # # First thing first: Spatio-temporal train test split

    # %%
    import stemflow
    isl_path = stemflow.__path__[0]
    print(f'Installation path: {isl_path}')
    
    #
    print('ST_train_test_split ...')
    from stemflow.model_selection import ST_train_test_split
    X_train, X_test, y_train, y_test = ST_train_test_split(X, y, 
                                                        Spatio_blocks_count = 100, Temporal_blocks_count=100,
                                                        random_state=42, test_size=0.3)
    print('Done.')

    # %% [markdown]
    # # Train AdaSTEM hurdle model

    # %%
    print('Importing stemflow modules...')
    from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
    from xgboost import XGBClassifier, XGBRegressor
    from stemflow.model.Hurdle import Hurdle_for_AdaSTEM, Hurdle
    print('Done.')

    # %%
    print('Declaring model instance...')
    
    fold_ = int(5 * (1/speed_up_times))
    min_req = min([1, int(fold_*0.7)])
    
    model = AdaSTEMRegressor(
        base_model=Hurdle(
            classifier=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
            regressor=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1)
        ),
        save_gridding_plot = True,
        ensemble_fold=fold_, 
        min_ensemble_required=min_req,
        grid_len_lon_upper_threshold=40,
        grid_len_lon_lower_threshold=5,
        grid_len_lat_upper_threshold=40,
        grid_len_lat_lower_threshold=5,
        temporal_start=1, 
        temporal_end=366,
        temporal_step=30,
        temporal_bin_interval=60,
        points_lower_threshold=30,
        Spatio1='longitude',
        Spatio2 = 'latitude', 
        Temporal1 = 'DOY',
        use_temporal_to_train=True,
        ensemble_models_disk_saver = ensemble_models_disk_saver,
        ensemble_models_disk_saving_dir = ensemble_models_disk_saving_dir,
        njobs=1                  
    )

    print('Done.')


    # %%
    print('Fitting model...')
    model.fit(X_train.reset_index(drop=True), y_train, verbosity=1)
    print('Done.')
    # %% [markdown]
    # # Feature importances

    # %%
    # Calcualte feature importance. This method is automatically called when fitting the model.
    # However, to show the process, we call it again.
    print('Calculating feature importances...')
    model.calculate_feature_importances()
    # print(model.feature_importances_)
    # stixel-specific feature importance is saved in model.feature_importances_
    print('Done.')

    # %%
    # Assign the feature importance to spatio-temporal points of interest
    print('Assigning importance to points...')
    importances_by_points = model.assign_feature_importances_by_points(verbosity=1, njobs=1)
    print('Done.')

    # %%
    # importances_by_points.head()

    # %%
    # top 10 important variables
    top_10_important_vars = importances_by_points[[
        i for i in importances_by_points.columns if not i in ['DOY','longitude','latitude','longitude_new','latitude_new']
        ]].mean().sort_values(ascending=False).head(10)

    print(top_10_important_vars)


    # %% [markdown]
    # ## Ploting the feature importances by variable names

    # %%
    from stemflow.utils.plot_gif import make_sample_gif

    # make spatio-temporal GIF for top 3 variables
    print('Plotting top 2 important variables...')
    for var_ in top_10_important_vars.index[:2]:
        print(f'Plotting {var_}...')
        make_sample_gif(importances_by_points, f'./stemflow_mini_test/FTR_IPT_{var_}.gif',
                                    col=var_, log_scale = False,
                                    Spatio1='longitude', Spatio2='latitude', Temporal1='DOY',
                                    figsize=(18,9), 
                                    xlims=(data.longitude.min()-10, data.longitude.max()+10), 
                                    ylims=(data.latitude.min()-10,data.latitude.max()+10), grid=True,
                                    xtick_interval=(data.longitude.max() - data.longitude.min())/8, 
                                    ytick_interval=(data.longitude.max() - data.longitude.min())/8,
                                    lng_size = 360, lat_size = 180, dpi=100, fps=10)
        
    print('Done.')

    # %% [markdown]
    # ![GIF of feature importance for variable `slope_mean`](../FTR_IPT_slope_mean.gif)

    # %% [markdown]
    # ## Plot uncertainty (error) in training 

    # %%
    # calculate mean and standard deviation in occurrence estimation
    print('Calculating the fitting errors...')
    pred_mean, pred_std = model.predict(X_train.reset_index(drop=True), 
                                                return_std=True, verbosity=1, njobs=1)

    print('Done.')

    # %%
    # Aggregate error to hexagon
    error_df = X_train[['longitude', 'latitude']]
    error_df.columns = ['lng', 'lat']
    error_df['pred_std'] = pred_std

    H_level = 3
    error_df = error_df.h3.geo_to_h3(H_level)
    error_df = error_df.reset_index(drop=False).groupby(f'h3_0{H_level}').mean()
    error_df = error_df.h3.h3_to_geo_boundary()



    # %%
    # plot mean error in hexagon
    error_df.plot('pred_std', legend=True, legend_kwds={'shrink':0.7})
    plt.grid(alpha=0.3)
    plt.title('Standard deviation in estimated mean occurrence')
    plt.savefig('./stemflow_mini_test/error_plot.pdf')
    
    if show:
        plt.show()
    else:
        plt.close()

    # %% [markdown]
    # # Evaluation

    # %%
    print('Predicting on test set...')
    pred = model.predict(X_test)
    print('Done.')

    # %%
    perc = np.sum(np.isnan(pred.flatten()))/len(pred.flatten())
    print(f'Percentage not predictable {round(perc*100, 2)}%')

    # %%
    pred_df = pd.DataFrame({
        'y_true':y_test.flatten(),
        'y_pred':np.where(pred.flatten()<0, 0, pred.flatten())
    }).dropna()


    # %%
    print('Evaluation...')
    print(AdaSTEM.eval_STEM_res('hurdle', pred_df.y_true, pred_df.y_pred))
    print('Done.')

    # %% [markdown]
    # # Plot QuadTree ensembles

    # %%
    
    if show:
        model.gridding_plot.show()


    # %%
    from watermark import watermark
    print(watermark())
    print(watermark(packages="stemflow,numpy,scipy,pandas,xgboost,tqdm,matplotlib,h3pandas,geopandas,scikit-learn,watermark"))


    # %%
    print('All Pass! ')
    if delet_tmp_files:
        print('Deleting tmp files...')
        import shutil
        shutil.rmtree('./stemflow_mini_test')
    
    print('Finish!')
    if show:
        return model.gridding_plot




