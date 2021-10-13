# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os
import warnings
from typing import Mapping, Optional, Sequence, Type, Union

from datasets.arrow_dataset import Dataset
from datasets.builder import DatasetBuilder
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.features import Features
from datasets.iterable_dataset import IterableDataset
from datasets.load import prepare_module
from datasets.metric import Metric
from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, hash_python_lines
from datasets.splits import Split
from datasets.streaming import extend_module_for_streaming
from datasets.tasks.base import TaskTemplate
from datasets.utils.download_manager import GenerateMode
from datasets.utils.file_utils import DownloadConfig
from datasets.utils.info_utils import is_small_dataset
from datasets.utils.version import Version
from omegaconf.dictconfig import DictConfig


def set_ignore_warnings():
    warnings.simplefilter("ignore")
    # set os environ variable for multiprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"


def import_main_class(module, dataset=True) -> Optional[Union[Type[DatasetBuilder], Type[Metric]]]:

    if dataset:
        main_cls_type = DatasetBuilder
    else:
        main_cls_type = Metric

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            break

    return module_main_cls


def load_dataset_builder(
    dataset_module,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[GenerateMode] = None,
    script_version: Optional[Union[str, Version]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    **config_kwargs,
) -> DatasetBuilder:
    # Download/copy dataset processing script
    _, hash, base_path = prepare_module(
        dataset_module.__file__,
        script_version=script_version,
        download_config=download_config,
        download_mode=download_mode,
        dataset=True,
        return_associated_base_path=True,
        use_auth_token=use_auth_token,
        data_files=data_files,
    )
    hash = hash_python_lines([hash, script_version]) # rehasing and consider script version
    builder_cls = import_main_class(dataset_module) # import the class from our own file

    # Instantiate the dataset builder
    builder_instance: DatasetBuilder = builder_cls(
        cache_dir=cache_dir,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        hash=hash,
        base_path=base_path,
        features=features,
        use_auth_token=use_auth_token,
        **config_kwargs,
    )

    return builder_instance


def load_my_dataset(
    dataset_module,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[GenerateMode] = None,
    ignore_verifications: bool = False,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    script_version: Optional[Union[str, Version]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    task: Optional[Union[str, TaskTemplate]] = None,
    streaming: bool = False,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """This is a modified version of datasets.load_dataset(), for easier debugging and configuration.
    """
    ignore_verifications = ignore_verifications or save_infos

    path = dataset_module.__file__
    # Create a dataset builder
    builder_instance = load_dataset_builder(
        dataset_module,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        features=features,
        download_config=download_config,
        download_mode=download_mode,
        script_version=script_version,
        use_auth_token=use_auth_token,
        **config_kwargs,
    )

    # Return iterable dataset in case of streaming
    if streaming:
        # this extends the open and os.path.join functions for data streaming
        extend_module_for_streaming(builder_instance.__module__, use_auth_token=use_auth_token)
        return builder_instance.as_streaming_dataset(
            split=split,
            use_auth_token=use_auth_token,
        )

    # Some datasets are already processed on the HF google storage
    # Don't try downloading from google storage for the packaged datasets as text, json, csv or pandas
    try_from_hf_gcs = path not in _PACKAGED_DATASETS_MODULES

    # Download and prepare data
    builder_instance.download_and_prepare(
        download_config=download_config,
        download_mode=download_mode,
        ignore_verifications=ignore_verifications,
        try_from_hf_gcs=try_from_hf_gcs,
        use_auth_token=use_auth_token,
    )

    # Build dataset for splits
    keep_in_memory = (
        keep_in_memory if keep_in_memory is not None else is_small_dataset(builder_instance.info.dataset_size)
    )
    ds = builder_instance.as_dataset(split=split, ignore_verifications=ignore_verifications, in_memory=keep_in_memory)
    # Rename and cast features to match task schema
    if task is not None:
        ds = ds.prepare_for_task(task)
    if save_infos:
        builder_instance._save_infos()

    return ds


def validate_resume_path(cfg: DictConfig):
    if not os.path.isfile(cfg.trainer.resume_from_checkpoint):
        cfg.trainer.resume_from_checkpoint = None
        
    return cfg
