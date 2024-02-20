"""
This is the top-level module for tracktrain

ModelSet and ModelDir are abstractions on the directory structure that is
enforced in the tracktrain framework.

ModelSet is a generalization of a number of models that share a parent
directory, and facilitates bulk access to all of the models' data.
This includes training metrics, configuration, and the models themselves.

ModelDir is a generalization of a single model's directory structure. Model
directories are created when a model is successfully compiled by
compile_and_build_dir, and minimally contain a _summary.txt and _config.json
"""
import numpy as np
import json
from pathlib import Path
from itertools import chain
from collections.abc import Callable
import tensorflow as tf

from tracktrain import utils
import tracktrain.model_methods as mm
from tracktrain.compile_and_train import compile_and_build_dir, train
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.config import compile_valid_args, compile_arg_descriptions
from tracktrain.config import train_valid_args, train_arg_descriptions
from tracktrain.config import compile_arg_defaults, train_arg_defaults

"""
Dict mapping string model abbreviations to methods that take a configuration
dictionary as a positional parameter and produce a valid Model object.
"""
model_builders = {
        "vae":VariationalEncoderDecoder.from_config,
        "ff":mm.feedforward_from_config,
        }

class ModelDir:
    """
    ModelDir is an abstraction for a directory minimally containing a config
    file sufficient to create a compilable Model object, and provides methods
    for interfacing with the model's configuration, training metrics, and
    trained weights.

    typical directory structure after training:

    model_parent_dir/
    |- model-0/
    |  |- model-0_config.json
    |  |- model-0_summary.txt
    |  |- model-0_prog.csv
    |  |- model-0_final.weights.hdf5
    |  |- model-0.keras
    |- model-1/
    |  |- model-1_config.json
    |  |- model-1_summary.txt
    |  |- model-1_prog.csv
    |  |- model-1_final.weights.hdf5
    |- model-2/
    |  |- model-2_config.json
    |  |- model-2_summary.txt
    |  |- model-2_prog.csv
    |  |- model-2_final.weights.hdf5
    """
    @staticmethod
    def build_from_config(config, model_parent_dir:Path, print_summary=True):
        """
        Initialize a model according to the configured model_type, which must
        be one of the keys in tracktrain.ModelDir.model_builders.

        This method executes tracktrain.compile_and_train.compile_and_build_dir
        and verifies that the subsetquent model directory is formatted right.

        Ultimately, returns the compiled Model object and corresponding
        ModelDir object as a 2-tuple.

        This method essentially serves as a wrapper on compile_and_build_dir
        that returns a ModelDir object in place of just the directory Path

        :@param config: dict containing model_type, and all the keys needed
            to initialize
        """
        model_type = config.get("model_type")
        if model_type is None or model_builders.get(model_type) is None:
            raise ValueError(
                    f"Config must contain model_type in ",
                    list(model_builders.keys())
                    )
        model = model_builders[model_type](config)
        model,model_dir_path = compile_and_build_dir(
                model=model,
                model_parent_dir=model_parent_dir,
                compile_config=config,
                )
        md = ModelDir(model_dir_path)
        md._check_req_files()
        return model,md

    def __init__(self, model_dir:Path):
        self.name = model_dir.name
        self.dir = model_dir
        children = [
                f for f in model_dir.iterdir()
                if (f.name.split("_")[0]==self.name and not f.is_dir())
                ]
        self.req_files = (
                self.dir.joinpath(f"{self.name}_summary.txt"),
                self.dir.joinpath(f"{self.name}_config.json"),
                )
        self.path_summary,self.path_config = self.req_files
        self.path_prog = self.dir.joinpath(f"{self.name}_prog.csv")
        ## final model should be stored as either weights or a serialized Model
        self.path_final_weights = self.dir.joinpath(
                f"{self.name}_final.weights.hdf5")
        self.path_final_model = self.dir.joinpath(
                f"{self.name}_final.hdf5")
        self._check_req_files()
        self._prog = None
        self._summary = None
        self._config = None
        self._final = None

    def summary(self):
        """ Return the {model}_summary.txt file as a string """
        if self._summary is None:
            self._summary = self.path_summary.open("r").read()
        return self._summary

    def prog(self):
        """
        Return the progress csv as a 2-tuple (labels:list, data:ndarray)
        data is a (E,M) array of M metrics' data per epoch E; each metric is
        labeled by the corresponding string in 'labels'.
        """
        if self._prog is None:
            self._prog = self.load_prog(as_array=True)
        return self._prog

    def final(self):
        """
        Loads and returns the "final" model or model weights if they exist
        in the model directory, None otherwise.

        It's up to the user to initialize the Model if there are only model
        weights available in the directory.

        Check the state of a ModelDir instance's weights config with:
        model_dir.config.get("save_weights_only")

        The final model file is identified by its name:
        {model}_final.hdf5  (OR)  {model}_final.weights.hdf5
        """
        if self._final is None:
            ## If not yet loaded,
            if self.path_final_model.exists():
                self._final = tf.keras.models.load_model(self.path_final_model)
            elif self.path_final_weights.exists():
                self._final = tf.keras.models.load_weights(
                        self.path_final_weights)
        return self._final

    @property
    def metric_labels(self):
        return self.prog()[0]
    @property
    def metric_data(self):
        """
        Returns a (E,M) shaped ndarray of M metric values over E epochs.
        """
        return self.prog()[1]
    @property
    def config(self):
        """ Returns the model config dictionary.  """
        if self._config == None:
            self._config = self._load_config()
        return self._config

    def get_metric(self, metric):
        """"
        Returns the per-epoch metric data array for one or more metrics.

        If a single str metric label is provided, a (E,) array of that metric's
        data over E epochs is returned.

        If a list of M str metric labels is provided, a (E,M) array of the
        corresponding metrics' data are provided (in the order of the labels).

        :@param metric: String metric label or list of metrics
        """
        if type(metric) is str:
            assert metric in self.metrics
            return self.metric_data[:,self.metric_labels.index(metric)]
        assert all(m in self.metric_labels for m in metric)
        idxs = np.array([self.metric_labels.index(m) for m in metric])
        return self.metric_data[:,idxs]

    def _check_req_files(self):
        """
        Verify that all files created by the build() function exist in the
        model directory (ie _summary.txt and _config.json).
        """
        try:
            assert all(f.exists() for f in self.req_files)
        except:
            raise FileNotFoundError(
                f"All of these files must be in {self.dir.as_posix()}:\n",
                tuple(f.name for f in self.req_files))
        return True

    def load_prog(self, as_array=False):
        """
        Load the training progress csv from a keras CSVLogger

        :@param: if True, loads progress lists as a single (E,M) ndarray
            for E epochs evaluated with M metrics
        """
        if self.path_prog is None:
            raise ValueError(
                    "Cannot return progress csv. "
                    f"File not found: {self.path_prog.as_posix()}"
                    )
        return utils.load_csv_prog(self.path_prog, as_array=as_array)

    def _load_config(self):
        """
        Load the configuration JSON associated with a specific model as a dict.
        """
        self._config = json.load(self.path_config.open("r"))
        return self._config

    def update_config(self, update_dict:dict):
        """
        Update the config json to have the new keys, replacing any that exist.

        Overwrites and reloads the json configuration file.

        :@param update_dict: dict mapping string config field labels to new
            json-serializable values.

        :@return: the config dict after being serialized and reloaded
        """
        ## Get the current configuration and update it
        cur_config = self.config
        cur_config.update(update_dict)
        ## Overwrite the json with the new version
        json.dump(cur_config, self.path_config.open("w"))
        ## reset the config and reload the json by returning the property
        self._config = None
        return self.config

class ModelSet:
    """
    A ModelSet abstracts collection of model instances
    """
    @staticmethod
    def from_dir(model_parent_dir:Path):
        """
        Assumes every subdirectory of the provided Path is a ModelDir-style
        model directory
        """
        model_dirs = [
                ModelDir(d) for d in model_parent_dir.iterdir() if d.is_dir()
                ]
        return ModelSet(model_dirs=model_dirs)

    def __init__(self, model_dirs:list, check_valid=True):
        """ """
        ## Validate all ModelDir objects unless check_valid is False
        assert check_valid or all(m._check_req_files() for m in model_dirs)
        self._models = tuple(model_dirs)

    @property
    def models(self):
        """ return the model directories as a tuple """
        return self._models
    @property
    def model_names(self):
        """ Return the string names of all ModelDir objects in the ModelSet """
        return tuple(m.name for m in self.models)

    def subset(self, rule:Callable=None, substr:str=None, check_valid=True):
        """
        Return a subset of the ModelDir objects in this ModelSet based on
        one or both of:

        (1) A Callable taking the ModelDir object and returning True or False.
        (2) A substring that must be included in the model dir's name property.

        :@param rule: Function taking a ModelDir as the first positional arg,
            and returning True iff the ModelDir should be in the new ModelSet
        :@param substr: String that must be included in the ModelDir.name
            string property of all directories in the returned ModelSet

        :@return: ModelSet with all ModelDir objects meeting the conditions
        """
        subset = self.models
        if not rule is None:
            subset = tuple(filter(rule, subset))
        if not substr is None:
            subset = tuple(filter(lambda m:substr in m.name, subset))
        return ModelSet(subset, check_valid=check_valid)

if __name__=="__main__":
    model_parent_dir = Path("/home/krttd/uah/24.s/aes690/aes690hw3/data/models")
    MS = ModelSet.from_dir(model_parent_dir)

    is_masked = lambda m:all(
            type(v) is float and v>0 for v in
            (m.config.get(l) for l in ("mask_pct", "mask_pct_stdev"))
            )
    sub = MS.subset(rule=is_masked) ## Models where masking was used
    #sub = MS.subset(substr="ved") ## Variational encoder-decoders

    print(list(sorted(m.name for m in sub.models)))
    print([m.metric_labels for m in sub.models])
    print([list(m.config.keys()) for m in sub.models])
    print([list(m.metric_data.shape) for m in sub.models])

    metric_labels,metric_data = zip(*[
        ml.load_prog(as_array=True) for ml in sub.models
        ])
    configs = [ml._load_config() for ml in sub.models]
    metric_union = set(chain(*metric_labels))
    metric_intersection = [
            m for m in metric_union
            if all(m in labels for labels in metric_labels)
            ]
    print(metric_union)
    print(metric_intersection)
