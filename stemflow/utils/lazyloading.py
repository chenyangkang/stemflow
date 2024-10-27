import os
import joblib
from collections.abc import MutableMapping


class LazyLoadingEnsembleDict(MutableMapping):
    def __init__(self, directory='./tmp_models'):
        """
        Initialize the LazyLoadingEnsembleDict with a directory to save and load models.
        
        Args:
            directory:
                The directory to save and load model files.
        """
        self.directory = directory
        self.ensemble_models = {}
        self.key_to_ensemble = {}
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=False)
        self._build_key_index()

    def _build_key_index(self):
        """
        Build an index mapping keys to ensemble IDs from saved ensemble files.
        """
        for filename in os.listdir(self.directory):
            if filename.startswith('ensemble_') and filename.endswith('_dict.pkl'):
                ensemble_id = filename[len('ensemble_'):-len('_dict.pkl')]
                ensemble_path = os.path.join(self.directory, filename)
                # Load the ensemble to get the keys
                ensemble = joblib.load(ensemble_path)
                for key in ensemble:
                    self.key_to_ensemble[key] = ensemble_id

    def _get_ensemble_id(self, key):
        """
        Extract the ensemble ID from the key.
        """
        if not isinstance(key, str):
            raise ValueError("Key must be a string.")
        parts = key.split('_')
        if len(parts) < 2:
            raise ValueError(f"Key '{key}' does not contain an ensemble ID.")
        return parts[0]

    def __getitem__(self, key):
        ensemble_id = self.key_to_ensemble.get(key)
        if not ensemble_id:
            raise KeyError(key)
        if ensemble_id not in self.ensemble_models:
            self.load_ensemble(ensemble_id)
        
        if key not in self.ensemble_models[ensemble_id]:
            if self.check_file_exists(ensemble_id):
                self.load_ensemble(ensemble_id, force=True)
            else:
                raise ValueError(f'Ensemble {key} (ensemble {ensemble_id}) not found in memory nor on disk.')
            
        return self.ensemble_models[ensemble_id][key]

    def __setitem__(self, key, value):
        ensemble_id = self._get_ensemble_id(key)
        if ensemble_id not in self.ensemble_models:
            self.ensemble_models[ensemble_id] = {}
        self.ensemble_models[ensemble_id][key] = value
        self.key_to_ensemble[key] = ensemble_id

    def __delitem__(self, key):
        ensemble_id = self.key_to_ensemble.get(key)
        if not ensemble_id:
            raise KeyError(key)
        if ensemble_id not in self.ensemble_models:
            self.load_ensemble(ensemble_id)
        del self.ensemble_models[ensemble_id][key]
        del self.key_to_ensemble[key]
        if not self.ensemble_models[ensemble_id]:
            del self.ensemble_models[ensemble_id]

    def __iter__(self):
        return iter(self.key_to_ensemble)

    def __len__(self):
        return len(self.key_to_ensemble)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self.keys())})"

    def __contains__(self, key):
        return key in self.key_to_ensemble

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self.key_to_ensemble.keys()

    def values(self):
        for key in self:
            yield self[key]

    def items(self):
        for key in self:
            yield (key, self[key])

    def pop(self, key, default=None):
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is not None:
                return default
            else:
                raise

    def clear(self):
        self.ensemble_models.clear()
        self.key_to_ensemble.clear()
        for filename in os.listdir(self.directory):
            if filename.startswith('ensemble_') and filename.endswith('_dict.pkl'):
                os.remove(os.path.join(self.directory, filename))

    def update(self, *args, **kwargs):
        for mapping in args:
            for key, value in mapping.items():
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def copy(self):
        new_copy = LazyLoadingEnsembleDict(self.directory)
        new_copy.ensemble_models = self.ensemble_models.copy()
        new_copy.key_to_ensemble = self.key_to_ensemble.copy()
        return new_copy

    def dump_ensemble(self, ensemble_id):
        """
        Dump the ensemble to disk and remove it from memory.
        """
        ensemble_id = str(ensemble_id)
        if ensemble_id in self.ensemble_models:
            ensemble_path = os.path.join(self.directory, f"ensemble_{ensemble_id}_dict.pkl")
            if not os.path.exists(ensemble_path):
                joblib.dump(self.ensemble_models[ensemble_id], ensemble_path)
            del self.ensemble_models[ensemble_id]
        else:
            raise ValueError(f'Ensemble {ensemble_id} does not exist in the current dictionary')

    def load_model(self, key):
        """
        Load the model corresponding to the key from disk into memory.
        """
        ensemble_id = self.key_to_ensemble.get(key)
        if not ensemble_id:
            raise KeyError(f"Key '{key}' not found.")
        self.load_ensemble(ensemble_id)
        
    def check_file_exists(self, ensemble_id):
        ensemble_path = os.path.join(self.directory, f"ensemble_{ensemble_id}_dict.pkl")
        if os.path.exists(ensemble_path):
            return True
        else:
            return False

    def load_ensemble(self, ensemble_id, force=False):
        """
        Load the entire ensemble from disk into memory.
        """
        ensemble_id = str(ensemble_id)
        if ((not force) and (ensemble_id not in self.ensemble_models)) or force:
            ensemble_path = os.path.join(self.directory, f"ensemble_{ensemble_id}_dict.pkl")
            if not os.path.exists(ensemble_path):
                raise FileNotFoundError(f"Ensemble file for ID {ensemble_id} not found at {ensemble_path}.")
            
            loaded_ensemble = joblib.load(ensemble_path)
            if ensemble_id in self.ensemble_models:
                loaded_ensemble = {**loaded_ensemble, **self.ensemble_models[ensemble_id]}
            
            self.ensemble_models[ensemble_id] = loaded_ensemble
            
            # Update key_to_ensemble mapping
            for key in self.ensemble_models[ensemble_id]:
                self.key_to_ensemble[key] = ensemble_id

    def delete_ensemble(self, ensemble_id):
        """
        Delete an ensemble from memory and disk.
        """
        ensemble_id = str(ensemble_id)
        if ensemble_id in self.ensemble_models:
            # Remove keys from key_to_ensemble
            for key in self.ensemble_models[ensemble_id]:
                del self.key_to_ensemble[key]
            del self.ensemble_models[ensemble_id]
        
        ensemble_path = os.path.join(self.directory, f"ensemble_{ensemble_id}_dict.pkl")
        if os.path.exists(ensemble_path):
            os.remove(ensemble_path)
        else:
            raise ValueError(f'Ensemble {ensemble_id} not found on disk.')
