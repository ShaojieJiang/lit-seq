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
from functools import partial
from importlib import import_module
from typing import Callable, Optional

from datasets import Dataset, load_dataset
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from transformers import default_data_collator

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig


class LanguageModelingDataModule(HFDataModule):
    """Defines ``LightningDataModule`` for Language Modeling Datasets.

    Args:
        *args: ``HFDataModule`` specific arguments.
        cfg: Contains data specific parameters when processing/loading the dataset
            (Default ``LanguageModelingDataConfig``)
        **kwargs: ``HFDataModule`` specific arguments.
    """

    cfg: LanguageModelingDataConfig

    def __init__(self, *args, cfg: LanguageModelingDataConfig = LanguageModelingDataConfig(), **kwargs) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)

    def load_dataset(self) -> Dataset:
        # Allow custom data files when loading the dataset
        data_files = {}
        if self.cfg.train_file is not None:
            data_files["train"] = self.cfg.train_file
        if self.cfg.validation_file is not None:
            data_files["validation"] = self.cfg.validation_file
        if self.cfg.test_file is not None:
            data_files["test"] = self.cfg.test_file

        data_files = data_files if data_files else None
        if self.cfg.dataset_name is not None:
            # Download and load the Huggingface dataset.
            dataset_module = import_module(f'..datasets.{self.cfg.dataset_name}', self.__module__)
            return load_dataset(
                path=dataset_module.__file__,
                name=self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                data_files=data_files,
            )

        # Load straight from data files
        if not data_files:
            raise MisconfigurationException(
                "You have not specified a dataset name. A custom train and validation file is required"
            )
        extension = self.cfg.train_file.split(".")[-1]
        return load_dataset(extension, data_files=data_files)

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        column_names = dataset["train" if stage == "fit" else "validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenize_function = partial(self.tokenize_function, tokenizer=self.tokenizer, text_column_name=text_column_name)
            
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        convert_to_features = partial(self.convert_to_features, block_size=self.effective_block_size, stride=self.cfg.stride)

        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        return dataset

    @property
    def effective_block_size(self) -> int:
        if self.cfg.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                log.warn(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({self.tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing dataset.cfg.block_size=x."
                )
            block_size = 1024
        else:
            if self.cfg.block_size > self.tokenizer.model_max_length:
                log.warn(
                    f"The block_size passed ({self.cfg.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.cfg.block_size, self.tokenizer.model_max_length)
        return block_size

    @staticmethod
    def tokenize_function(
        examples,
        tokenizer,
        text_column_name: str = None,
    ):
        return tokenizer(examples[text_column_name])

    @staticmethod
    def convert_to_features(examples, block_size: int, stride: int, **kwargs):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, stride)]
            for k, t in concatenated_examples.items()
        }
        if len(result['input_ids'][-1]) < block_size:
            result['input_ids'] = result['input_ids'][:-1]
            result['attention_mask'] = result['attention_mask'][:-1]
        result["labels"] = result["input_ids"].copy()
        return result

    @property
    def collate_fn(self) -> Callable:
        return default_data_collator
