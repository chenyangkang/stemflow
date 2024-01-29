from multiprocessing import Lock, Process, shared_memory

import numpy as np
import pandas as pd


def mp_training_func(ensemble, instance, data):
    return instance.SAC_ensemble_training(index_df=ensemble[1], data=data)


def mp_predict_func(ensemble, instance, data):
    return instance.SAC_ensemble_predict(index_df=ensemble[1], data=data)


def mp_make_shared_mem(X):
    X_data = X.values
    shm = shared_memory.SharedMemory(create=True, size=X_data.nbytes)
    np_array = np.ndarray(X_data.shape, dtype=np.float32, buffer=shm.buf)
    np_array[:] = X_data[:]
    return shm, np_array


def mp_predict_func_shr_mem(ensemble, instance, X_names, X_shape, lock, shr_name):
    existing_shm = shared_memory.SharedMemory(name=shr_name)
    np_array = np.ndarray(X_shape, dtype=np.float32, buffer=existing_shm.buf)
    data = pd.DataFrame(np_array, columns=X_names)
    print(data)

    lock.acquire()
    res = instance.SAC_ensemble_predict(index_df=ensemble[1], data=data)
    lock.release()
    existing_shm.close()

    return res
