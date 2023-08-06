import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def random_sample_with_N_digits(n, s):
    import random
    range_start = 10**(n-1)
    range_end = (10**n)-1
    candidates = range(range_start, range_end)
    return random.sample(candidates, s)

def random_sampling_event_identifier_generator(size=366, number_digit=7):
    return ['S'+str(i) for i in random_sample_with_N_digits(number_digit, size)]


def generate_fake_dataset(lat_grid_count = 18, lng_grid_count=36):
    lnglng, latlat = np.meshgrid(np.linspace(-170,170,lng_grid_count),
                                 np.linspace(-80,80,lat_grid_count))
    data = pd.DataFrame({
        # 'DOY': [random.randrange(1,367) for i in range(size)]
        'longitude':lnglng.flatten(),
        'latitude':latlat.flatten()
    })
    
    new_data = []
    for doy in range (1,367):
        tmp = data.copy()
        tmp['DOY'] = doy
        new_data.append(tmp)
        
    data = pd.concat(new_data, axis=0).reset_index(drop=True)
    data = data.sample(frac=0.7, replace=False) # mimic the incomplete sampling
    data['sampling_event_identifier'] = random_sampling_event_identifier_generator(len(data))
    
    
    ###
    betas = [np.random.uniform(-1,1) for i in range(5)]
    trait1 = betas[0]*(data['longitude']/20)**2
    std = np.std(trait1)
    error = np.random.normal(size=len(trait1)) * std * 0.05
    trait1 += error
    
    trait2 = betas[1]*(data['latitude']/20)**2
    std = np.std(trait2)
    error = np.random.normal(size=len(trait2)) * std * 0.05
    trait2 += error
    
    trait3 = betas[2] * np.cos(data['DOY']/20) 
    std = np.std(trait3)
    error = np.random.normal(size=len(trait3)) * std * 0.05
    trait3 += error
    
    ### this trait4 will be the hidden spatial effect that cannot be observed
    trait4= np.random.normal(loc=0, scale=0.1) * np.cos(data['longitude']/20 + np.random.normal(size=1)) + \
                np.random.normal(loc=0, scale=0.1) * (data['longitude']/20)**2
    trait5 = np.digitize(data['longitude'],
                         np.linspace(data['longitude'].min(), data['longitude'].max(), 10)) * trait3
    
    
    std = np.std(trait4)
    error = np.random.normal(size=len(trait4)) * std * 0.05
    trait4 += error
    
    r = trait1 + trait2 + 1e-8
    t = trait3
    
    ### abundance is a non-linear combination of traits 1, 2, and 3
    # abundance =  betas[2] * np.sin(r * t * np.pi) + betas[3] * np.cos(r * t)
    abundance =  np.sin(r * t * np.pi) + np.cos(r * t) + trait4 + trait5
    std_abundance = np.std(abundance)
    error = np.random.normal(size=len(abundance)) * std_abundance * 0.05
    abundance = abundance + error
                   
    data['trait1'] = trait1
    data['trait2'] = trait2
    data['trait3'] = trait3
    data['abundance'] = abundance
    data['abundance'] = np.where(data['abundance']<=0, 0, data['abundance'])
    
    total_zero_frac = np.sum(data['abundance']<=0) / len(data)
    if total_zero_frac>0.9:
        raise AttributeError('Dataset generation fail. Try rerun the generation.')
    elif total_zero_frac<0.3:
        ## how many should I add?
        how_many_to_add = int((0.4 - total_zero_frac) * len(data))
        the_index = np.random.choice(list(data[data['abundance']>0].index),
                         how_many_to_add,
                         replace=False)
        data['abundance'][data.index.isin(the_index)]= 0
        
    print('Positive count: ',len(data[data['abundance']>0]), '\nZero count:',len(data[data['abundance']==0]))
    return data


