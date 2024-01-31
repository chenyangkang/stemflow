import pickle
from functools import partial
from multiprocessing import Lock, Process, shared_memory

import joblib
import numpy as np
import pandas as pd


# Model things
def load_model_to_shr_mem(model):
    # Load the serialized model into shared memory
    model_bytes = pickle.dumps(model)
    model_shm = shared_memory.SharedMemory(create=True, size=len(model_bytes))
    model_array = np.frombuffer(model_shm.buf, dtype=np.uint8)
    model_array[: len(model_bytes)] = bytearray(model_bytes)
    return model_bytes, model_shm


def load_model_from_shr_mem(model_bytes, model_shm):
    model_memoryview = memoryview(model_shm.buf)[: len(model_bytes)]
    loaded_model = pickle.loads(model_memoryview)
    return loaded_model


# Data things
def load_data_to_shr_mem(data):
    # Load the serialized data into shared memory
    data_bytes = pickle.dumps(data)
    data_shm = shared_memory.SharedMemory(create=True, size=len(data_bytes))
    data_array = np.frombuffer(data_shm.buf, dtype=np.uint8)
    data_array[: len(data_bytes)] = bytearray(data_bytes)
    return data_bytes, data_shm


def load_data_from_shr_mem(data_bytes, data_shm):
    data_memoryview = memoryview(data_shm.buf)[: len(data_bytes)]
    loaded_data = pickle.loads(data_memoryview)
    return loaded_data


#
def mp_split_func_shr_mem(data_bytes, data_shm, partial_get_one_ensemble_quadtree):
    loaded_data = load_data_from_shr_mem(data_bytes, data_shm)
    partial_get_one_ensemble_quadtree = partial(partial_get_one_ensemble_quadtree, data=loaded_data)

    return partial_get_one_ensemble_quadtree


def mp_training_func_shr_mem(ensemble, model_bytes, model_shm, data_bytes, data_shm):
    loaded_model = load_model_from_shr_mem(model_bytes, model_shm)
    loaded_data = load_data_from_shr_mem(data_bytes, data_shm)

    res = loaded_model.SAC_ensemble_training(index_df=ensemble[1], data=loaded_data)

    return res


def mp_predict_func_shr_mem(ensemble, model_bytes, model_shm, data_bytes, data_shm):
    loaded_model = load_model_from_shr_mem(model_bytes, model_shm)
    loaded_data = load_data_from_shr_mem(data_bytes, data_shm)

    res = loaded_model.SAC_ensemble_predict(index_df=ensemble[1], data=loaded_data)

    return res
