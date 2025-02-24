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
import h5py
from tensorflow.python.keras.saving.hdf5_format \
        import load_weights_from_hdf5_group
import matplotlib.pyplot as plt

from tracktrain import utils
import tracktrain.model_methods as mm
import tracktrain.old_methods as om
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
        #"ved":VariationalEncoderDecoder.from_config,
        #"ff":mm.feedforward_from_config,
        #"paed_old":lambda c:om.get_paed_old(**c),
        #"paed_old2":lambda c:om.get_paed_old2(**c),
        #"paed":lambda c:om.get_paed(**c),
        "ceda":lambda c:mm.get_ceda(**c),
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
    def build_from_config(config, model_parent_dir:Path, print_summary=True,
            custom_model_builders={}, custom_losses={}, custom_metrics={},
            custom_lr_schedulers={}):
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
        :@param custom_losses: Dictionary of valid custom loss functions
        :@param custom_metrics: Dictionary of valid custom metric functions

        :@return: 2-tuple (model,md) where 'model' is a compiled Model instance
            and md is a ModelDir object associated with an existing directory
            for the associated model object.
        """
        model_type = config.get("model_type")
        tmp_model_builders = {**model_builders, **custom_model_builders}
        if model_type is None or tmp_model_builders.get(model_type) is None:
            raise ValueError(
                    f"When initializing a ModelDir, config must "
                    "provide model_type in ", list(tmp_model_builders.keys())
                    )
        model = tmp_model_builders[model_type](config.get("model"))
        model,model_dir_path = compile_and_build_dir(
                model=model,
                model_parent_dir=model_parent_dir,
                config=config,
                custom_losses=custom_losses,
                custom_metrics=custom_metrics,
                print_summary=print_summary,
                )
        md = ModelDir(model_dir_path, custom_model_builders=tmp_model_builders)
        md._check_req_files()
        return model,md

    def __init__(self, model_dir:Path, custom_model_builders={}):
        """ Initialize a ModelDir from an existing directory """
        if type(model_dir) == str:
            model_dir = Path(model_dir)
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
        self.model_builders = model_builders
        self.model_builders.update(custom_model_builders)
        self.path_summary,self.path_config = self.req_files
        self.path_prog = self.dir.joinpath(f"{self.name}_prog.csv")
        ## final model should be stored as either weights or a serialized Model
        self.path_final_weights = self.dir.joinpath(
                f"{self.name}_final.weights.h5")
        self.path_final_model = self.dir.joinpath(
                f"{self.name}_final.h5")
        self._check_req_files()
        self._prog = None
        self._summary = None
        self._config = None
        self._final_weights = None
        self._final_model = None

    def __str__(self):
        return f"ModelDir({self.dir})"
    def __repr__(self):
        return str(self)

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

    def load_model(self, model_path=None, debug=False):
        """
        Loads and returns the "final" model object if it exists in the
        model directory.

        Check the state of a ModelDir instance's weights config with:
        model_dir.config.get("save_weights_only")

        The final model model file is identified by its name:
        {model}_final.hdf5
        """
        if model_path is None:
            assert self.path_final_model.exists()
            model_path = self.path_final_model
        if debug:
            print(f"Loading {model_path.as_posix()}")
        return tf.keras.saving.load_model(model_path)

    def load_weights(self, weights_path:Path=None):
        """
        Initializes a model according to this ModelDir's configuration
        (which requires that the model_type field to be one of the
        model_builders  keys. Loads and returns the "final" model weights if
        they exist in the model directory, or loads model weights from the
        user-provided path

        It's up to the user to initialize the Model if there are only model
        weights available in the directory.

        Check the state of a ModelDir instance's weights config with:
        model_dir.config.get("save_weights_only")

        The final model weights file is identified by its name:
        {model}_final.weights.hdf5
        """
        #load_path = (model_path, self.path_final_weights)[model is None]
        #assert load_path.exists()
        if not weights_path is None:
            weights_path = Path(weights_path)
        if self.config.get("model_type") not in self.model_builders.keys():
            raise ValueError(
                    f"model_type={self.config.get('model_type')} must be in ",
                    list(self.model_builders.keys()))
        model = self.model_builders.get(
                self.config.get("model_type")
                )(self.config.get("model"))
        if weights_path is None:
            #print(self.path_final_weights)
            if not self.path_final_weights.exists():
                raise ValueError(
                        f"No final weights found in {self.dir.as_posix()}")
            weights_path = self.path_final_weights
        elif weights_path.name in (p.name for p in self.dir.iterdir()):
            weights_path = self.dir.joinpath(weights_path.name)
        try:
            model.load_weights(weights_path)
        except ValueError as e:
            print(f"IGNORING COMPATIBILITY ERROR:\n{e}")
            print(f"Loading using tensorflow 15 style")
            with h5py.File(weights_path, 'r') as f:
                load_weights_from_hdf5_group(f["layers"], model.layers)
        return model

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
            assert metric in self.metric_labels
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

    def load_prog(self, as_array=True):
        """
        Load the training progress csv from a keras CSVLogger as a 2-tuple
        like (labels, array) where labels is a list of unique strings, and
        array is a (E,M) shaped array with M metrics over E epochs.
        labels is a list of M unique strings labeling the M metrics.

        :@param as_array: if True, loads progress lists as a single (E,M)
            ndarray for E epochs evaluated with M metrics. Otherwise, returns
            the data as a list of 1d arrays.
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

        CAREFUL! Overwrites and reloads the json configuration file.

        This is useful for retroactively updating json files that must meet
        a newly-enforced standard, or for recategorizing models.

        :@param update_dict: dict mapping string config field labels to new
            json-serializable values.

        :@return: the config dict after being serialized and reloaded
        """
        ## Get the current configuration and update it
        cur_config = self.config
        cur_config.update(update_dict)
        ## Overwrite the json with the new version
        json.dump(cur_config, self.path_config.open("w"), indent=4)
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
                ModelDir(d)
                for d in Path(model_parent_dir).iterdir()
                if d.is_dir()
                ]
        return ModelSet(model_dirs=model_dirs)

    def __init__(self, model_dirs:list, check_valid=True):
        """ """
        assert all(type(m) is ModelDir for m in model_dirs)
        ## Validate all ModelDir objects unless check_valid is False
        assert check_valid or all(m._check_req_files() for m in model_dirs)
        self._model_dirs = tuple(model_dirs)

    def __str__(self):
        return ", ".join(list(map(str,self._model_dirs)))
    def __repr__(self):
        return str(self)

    @property
    def model_dirs(self):
        """ return the model directories as a tuple """
        return self._model_dirs
    @property
    def model_names(self):
        """ Return the string names of all ModelDir objects in the ModelSet """
        return tuple(m.name for m in self.model_dirs)

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
        subset = self.model_dirs
        if not rule is None:
            subset = tuple(filter(rule, subset))
        if not substr is None:
            subset = tuple(filter(lambda m:substr in m.name, subset))
        return ModelSet(subset, check_valid=check_valid)

    def scatter_metrics(self,xmetric:str,ymetric:str,fig_path=None,show=False,
                     use_notes=False, plot_spec={}):
        """  """
        ps = {"xlabel":"epoch", "ylabel":"", "title":"", "cmap":"Set1",
              "text_size":12, "norm":"linear", "logx":False, "figsize":(16,12),
              "plot_kwargs":{}, "xlim":None, "ylim":None, "line_width":2,
              "facecolor":"white", "legend_cols":1, "legend_size":8}
        #line_styles = ("-", ":", "--", "-.")
        ps = {**ps, **plot_spec}
        fig,ax = plt.subplots()

        model_xmetrics = []
        model_ymetrics = []
        model_labels = []
        for md in sorted(self.model_dirs,key=lambda m:m.dir.name):
            model_xmetrics.append(md.get_metric(xmetric)[-1])
            model_ymetrics.append(md.get_metric(ymetric)[-1])
            model_labels.append(f"{md.name}")
            if use_notes:
                model_labels[-1] += " - " + md.config.get("notes")

        print(len(model_labels))

        cmap = plt.cm.get_cmap(ps.get("cmap"), len(model_labels))
        print(len(model_xmetrics), len(model_ymetrics), len(model_labels))
        scatter = plt.scatter(
                model_xmetrics,
                model_ymetrics,
                c=list(range(len(model_labels))),
                cmap=cmap,
                )
        leg_elem = scatter.legend_elements(num=len(model_labels))
        ax.legend(handles=leg_elem[0],
                  ncols=ps.get("legend_cols"),
                  labels=model_labels,
                  prop={"size":ps.get("legend_size")},
                  )
        if ps["logx"]:
            plt.semilogx()

        ax.set_title(ps.get("title"))
        ax.set_xlabel(ps.get("xlabel"))
        ax.set_ylabel(ps.get("ylabel"))
        ax.set_facecolor(ps.get("facecolor"))
        if not ps.get("xlim") is None:
            ax.set_xlim(*ps["xlim"])
        if not ps.get("ylim") is None:
            ax.set_ylim(*ps["ylim"])
        if show:
            plt.show()
        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)

    def plot_metrics(self,metrics:list,fig_path=None,show=False,
                     use_notes=False, plot_spec={}):
        """  """
        ps = {"xlabel":"epoch", "ylabel":"", "title":"", "cmap":"Set1",
              "text_size":12, "norm":"linear", "logx":False, "figsize":(16,12),
              "plot_kwargs":{}, "xlim":None, "ylim":None, "line_width":2,
              "facecolor":"white", "legend_cols":1, "fontsize_legend":12,
              "fontsize_title":16, "fontsize_labels":12,}
        line_styles = ("-", ":", "--", "-.")
        ps = {**ps, **plot_spec}
        fig,ax = plt.subplots()
        cmap = plt.cm.get_cmap(ps.get("cmap"), len(self.model_dirs))
        for i,md in enumerate(sorted(self.model_dirs,key=lambda m:m.dir.name)):
            for j,m in enumerate(metrics):
                if m not in md.metric_labels:
                    raise ValueError(f"{md} doesn't support metric {m}")
                label = f"{md.name} - "
                if use_notes:
                    label += md.config.get("notes") + " - "
                label += m
                lineplot = ax.plot(
                        md.get_metric("epoch"),
                        md.get_metric(m),
                        linewidth=ps.get("line_width"),
                        label=label,
                        color=cmap(i),
                        linestyle=line_styles[j%len(line_styles)],
                        **ps.get("plot_kwargs"),
                        )
        legend = ax.legend(
                ncols=ps.get("legend_cols"),
                fontsize=ps.get("fontsize_legend"),
                )
        if ps["logx"]:
            plt.semilogx()
        ax.set_title(ps.get("title"), fontsize=ps.get("fontsize_title"))
        ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("fontsize_labels"))
        ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("fontsize_labels"))
        ax.set_facecolor(ps.get("facecolor"))
        if not ps.get("xlim") is None:
            ax.set_xlim(*ps["xlim"])
        if not ps.get("ylim") is None:
            ax.set_ylim(*ps["ylim"])
        if show:
            plt.show()
        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)


if __name__=="__main__":
    model_parent_dir = Path(
            "/home/krttd/uah/24.s/aes690/aes690hw3/data/models")
    MS = ModelSet.from_dir(model_parent_dir)

    is_masked = lambda m:all(
            type(v) is float and v>0 for v in
            (m.config.get(l) for l in ("mask_pct", "mask_pct_stdev"))
            )
    sub = MS.subset(rule=is_masked) ## Models where masking was used
    #sub = MS.subset(substr="ved") ## Variational encoder-decoders

    get_unique = lambda ll:tuple(set(chain(*ll)))
    ## print all of the model subdirectory names
    print(list(sorted(m.name for m in sub.model_dirs)))
    ## print list of all config keys in any model
    print(get_unique([list(m.config.keys()) for m in sub.model_dirs]))
    ## print list of all metric labels in any model
    print(get_unique([m.metric_labels for m in sub.model_dirs]))
    #print([list(m.metric_data.shape) for m in sub.model_dirs])

    ## Print all available metrics and metrics common between all models
    metric_labels,metric_data = zip(*[
        ml.load_prog(as_array=True) for ml in sub.model_dirs
        ])
    configs = [ml._load_config() for ml in sub.model_dirs]
    metric_union = set(chain(*metric_labels))
    metric_intersection = [
            m for m in metric_union
            if all(m in labels for labels in metric_labels)
            ]
    print(metric_union)
    print(metric_intersection)
