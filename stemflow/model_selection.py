from pandas.core.frame import DataFrame
from numpy import ndarray
import numpy as np
import pandas as pd
from .utils import check_random_state

def ST_train_test_split(X: DataFrame, y, 
                        Spatio1: str = 'longitude', Spatio2: str = 'latitude', Temporal1: str = 'DOY',
                        Spatio_blocks_count = 10,
                        Temporal_blocks_count = 10,
                        test_size = 0.3,
                        random_state = None,
                        ) -> (DataFrame, ndarray):
    """Spatial Temporal train-test split
    
    Parameters
    ----------
    X: DataFrame
    
    y: DataFrame or numpy array
    
    Spatio1: str
        column name of spatial indicator 1
        
    Spatio2: str
        column name of spatial indicator 2
        
    Temporal1: str
        column name of temporal indicator 1
        
    Spatio_blocks_count: int
        How many block to split for spatio indicators
        
    Temporal_blocks_count: int
        How many block to split for temporal indicators
    
    test_size: float
        Fraction of test set in terms of blocks count
        
    random_state: int
        random state for choosing testing blocks
        
    Returns
    ---------
    X_train: DataFrame
    X_test: DataFrame
    y_train: np.darray
    y_test: np.darray
    
    """
    # random seed
    rng = check_random_state(random_state)
    
    # validate
    if not isinstance(X, DataFrame):
        type_x = str(type(X))
        raise TypeError(f'X input should be pandas.core.frame.DataFrame, Got {type_x}')
    if not (isinstance(y, DataFrame) or isinstance(y, ndarray)):
        type_y = str(type(y))
        raise TypeError(f'y input should be pandas.core.frame.DataFrame or numpy.ndarray, Got {type_y}')
    
    # check shape match
    y_size = np.array(y).flatten().shape[0]
    X_size = X.shape[0]
    if not y_size == X_size:
        raise ValueError(f'The shape of X and y should match. Got X: {X_size}, y: {y_size}')
    
    # indexing
    Sindex1 = np.linspace(X[Spatio1].min(), X[Spatio1].max(), Spatio_blocks_count)
    Sindex2 = np.linspace(X[Spatio2].min(), X[Spatio2].max(), Spatio_blocks_count)
    Tindex1 = np.linspace(X[Temporal1].min(), X[Temporal1].max(), Temporal_blocks_count)
    
    indexes = [str(a)+'_'+str(b)+'_'+str(c) for a,b,c in zip(
        np.digitize(X[Spatio1],Sindex1),
        np.digitize(X[Spatio2],Sindex2),
        np.digitize(X[Temporal1],Tindex1)
    )]
    
    unique_indexes = list(np.unique(indexes))

    # get test set record indexes
    test_indexes = []
    test_cell = list(rng.choice(unique_indexes, replace=False, size=int(len(unique_indexes)*test_size)))

    for index, cell in enumerate(indexes):
        if cell in test_cell:
            test_indexes.append(index)
        
    # get train set record indexes
    train_indexes = list(set(range(len(indexes))) - set(test_indexes))

    # get train test data
    X_train = X.iloc[train_indexes, :]
    y_train = np.array(y).flatten()[train_indexes].reshape(-1,1)
    X_test = X.iloc[test_indexes, :]
    y_test = np.array(y).flatten()[test_indexes].reshape(-1,1)
    
    return X_train, X_test, y_train, y_test
    
    
    
    
def ST_CV(X: DataFrame, y, 
                        Spatio1: str = 'longitude', Spatio2: str = 'latitude', Temporal1: str = 'DOY',
                        Spatio_blocks_count = 10,
                        Temporal_blocks_count = 10,
                        random_state = None,
                        CV=3,
                        ):
    """Spatial Temporal train-test split
    
    Parameters
    ----------
    X: DataFrame
    
    y: DataFrame or numpy array
    
    Spatio1: str
        column name of spatial indicator 1
        
    Spatio2: str
        column name of spatial indicator 2
        
    Temporal1: str
        column name of temporal indicator 1
        
    Spatio_blocks_count: int
        How many block to split for spatio indicators
        
    Temporal_blocks_count: int
        How many block to split for temporal indicators
    
    test_size: float
        Fraction of test set in terms of blocks count
        
    random_state: int
        random state for choosing testing blocks
        
    CV: int
        fold cross validation
        
    Returns
    ---------
    X_train: DataFrame
    X_test: DataFrame
    y_train: np.darray
    y_test: np.darray
    
    """
    # random seed
    rng = check_random_state(random_state)
    
    # validate
    if not isinstance(X, DataFrame):
        type_x = str(type(X))
        raise TypeError(f'X input should be pandas.core.frame.DataFrame, Got {type_x}')
    if not (isinstance(y, DataFrame) or isinstance(y, ndarray)):
        type_y = str(type(y))
        raise TypeError(f'y input should be pandas.core.frame.DataFrame or numpy.ndarray, Got {type_y}')
    if not (isinstance(CV, int) and CV>0):
        raise ValueError('CV should be a positive interger')
    
    # check shape match
    y_size = np.array(y).flatten().shape[0]
    X_size = X.shape[0]
    if not y_size == X_size:
        raise ValueError(f'The shape of X and y should match. Got X: {X_size}, y: {y_size}')
    
    # indexing
    Sindex1 = np.linspace(X[Spatio1].min(), X[Spatio1].max(), Spatio_blocks_count)
    Sindex2 = np.linspace(X[Spatio2].min(), X[Spatio2].max(), Spatio_blocks_count)
    Tindex1 = np.linspace(X[Temporal1].min(), X[Temporal1].max(), Temporal_blocks_count)
    
    indexes = [str(a)+'_'+str(b)+'_'+str(c) for a,b,c in zip(
        np.digitize(X[Spatio1],Sindex1),
        np.digitize(X[Spatio2],Sindex2),
        np.digitize(X[Temporal1],Tindex1)
    )]
    
    unique_indexes = list(np.unique(indexes))
    rng.shuffle(unique_indexes)
    test_size = int(len(unique_indexes) * (1/CV))
    
    for cv_count in range(CV):
        # get test set record indexes
        test_indexes = []
        start = cv_count*test_size
        end = np.min([(cv_count+1)*test_size, len(unique_indexes)+1])
        test_cell = unique_indexes[start: end]

        for index, cell in enumerate(indexes):
            if cell in test_cell:
                test_indexes.append(index)
            
        # get train set record indexes
        train_indexes = list(set(range(len(indexes))) - set(test_indexes))

        # get train test data
        X_train = X.iloc[train_indexes, :]
        y_train = np.array(y).flatten()[train_indexes].reshape(-1,1)
        X_test = X.iloc[test_indexes, :]
        y_test = np.array(y).flatten()[test_indexes].reshape(-1,1)
        
        yield X_train, X_test, y_train, y_test
    
        
    
