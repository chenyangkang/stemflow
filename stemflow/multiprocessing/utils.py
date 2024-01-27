def mp_training_func(ensemble, instance, data):
    return instance.SAC_ensemble_training(index_df=ensemble[1], data=data)


def mp_predict_func(ensemble, instance, data):
    return instance.SAC_ensemble_predict(index_df=ensemble[1], data=data)
