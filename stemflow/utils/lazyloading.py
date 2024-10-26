import joblib
import os


class LazyLoadingEnsembleDict():
    def __init__(self, directory='./tmp_models'):
        """
        Initialize the DumpableDict with a directory to save and load models.
        :param directory: The directory to save and load model files.
        """
        self.directory = directory
        self.ensemble_models = {}
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=False)

    def __repr__(self):
        return str(self.ensemble_models)
    
    def __getitem__(self, key):
        # Load the model from disk if it has been dumped
        ensemble_id = key.split('_')[1]

        all_model_names = self.all_model_names()
        
        if key not in all_model_names:
            self.load_model(str(ensemble_id))

        return self.ensemble_models[str(ensemble_id)][key]

    def __setitem__(self, key, value):
        # Set the model in the dictionary
        ensemble_id = key.split('_')[1]
        if not ensemble_id in self.ensemble_models:
            self.ensemble_models[str(ensemble_id)] = {}
            
        self.ensemble_models[str(ensemble_id)][key] = value

    def all_model_names(self):
        all_model_names = []
        for i in self.ensemble_models:
            all_model_names.extend(self.ensemble_models[i])
        return all_model_names
    
    def dump_ensemble(self, ensemble_id):
        """
        Dump the model to disk and remove it from memory.
        """
        if str(ensemble_id) in self.ensemble_models:
            if os.path.exists(os.path.join({self.directory}, f"ensemble_{ensemble_id}_dict.pkl")):
                pass
            else:
                joblib.dump(self.ensemble_models[str(ensemble_id)], os.path.join({self.directory}, f"ensemble_{ensemble_id}_dict.pkl"))
            del self.ensemble_models[str(ensemble_id)]
        else:
            raise ValueError(f'Ensemble {str(ensemble_id)} not exist in the current dictionary')

    def load_model(self, key):
        """
        Load the model from disk into memory.
        """
        ensemble_id = key.split('_')[1]
        self.load_ensemble(str(ensemble_id))

    def load_ensemble(self, ensemble_id):
        """
        Load the whole ensemble of models from disk into memory.
        """
        if str(ensemble_id) not in self.ensemble_models:
            self.ensemble_models[str(ensemble_id)] = joblib.load(os.path.join({self.directory}, f"ensemble_{ensemble_id}_dict.pkl"))
