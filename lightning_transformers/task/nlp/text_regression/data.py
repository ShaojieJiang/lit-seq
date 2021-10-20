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
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional

from datasets import ClassLabel, Dataset
from datasets.load import load_dataset
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorWithPadding

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.core.utils import load_my_dataset


class TextRegressionDataModule(HFDataModule):
    """Defines the ``LightningDataModule`` for Text Regression Datasets."""

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        input_feature_fields = [k for k, v in dataset["train"].features.items() if k not in ["label", "idx"]]
        dataset = self.__class__.preprocess(
            dataset,
            tokenizer=self.tokenizer,
            input_feature_fields=input_feature_fields,
            padding=self.cfg.padding,
            truncation=self.cfg.truncation,
            max_length=self.cfg.max_length,
        )
        cols_to_keep = [
            x for x in ["input_ids", "attention_mask", "token_type_ids", "labels", "dialog_id", "turn_id"] if x in dataset["train"].features
        ]
        # if not isinstance(dataset["train"].features["labels"], ClassLabel):
        #     dataset = dataset.class_encode_column("labels")

        dataset.set_format("torch", columns=cols_to_keep)
        return dataset

    @property
    def model_data_kwargs(self) -> Dict[str, int]:
        return {"num_labels": 1} # for regression we need only 1 class

    @staticmethod
    def convert_to_features(
        example_batch: Any, _, tokenizer: PreTrainedTokenizerBase, input_feature_fields: List[str], **tokenizer_kwargs
    ):
        texts_or_text_pairs = example_batch[input_feature_fields[0]]
        # Tokenize the text/text pairs
        return tokenizer(texts_or_text_pairs, **tokenizer_kwargs)

    @staticmethod
    def preprocess(ds: Dataset, **fn_kwargs) -> Dataset:
        ds = ds.map(
            # todo: change this to self.convert_to_features for users to override
            TextRegressionDataModule.convert_to_features,
            batched=True,
            with_indices=True,
            fn_kwargs=fn_kwargs,
        )
        ds.rename_column_("label", "labels")
        return ds

    def load_dataset(self) -> Dataset:
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
            try:
                dataset_module = import_module(f'..datasets.{self.cfg.dataset_name}', self.__module__)
                return load_my_dataset(
                    dataset_module,
                    name=self.cfg.dataset_config_name,
                    cache_dir=self.cfg.cache_dir,
                    data_files=data_files,
                    history_delimeter=self.cfg.history_delimeter,
                    history_size=self.cfg.history_size,
                    script_version=f'histsz_{self.cfg.history_size}',
                    hierarchical=self.cfg.hierarchical,
                )
            except: # not a customised dataset
                return load_dataset(
                    path=self.cfg.dataset_name,
                    name=self.cfg.dataset_config_name,
                    cache_dir=self.cfg.cache_dir,
                    data_files=data_files,
                )
    
    @property
    def collate_fn(self) -> Optional[Callable]:
        if self.cfg.padding != 'max_length':
            return DataCollatorWithPadding(self.tokenizer)
        else:
            return super().collate_fn


class TextRegressionMultiDataModule(TextRegressionDataModule):
    """Defines the ``LightningDataModule`` for Text Regression Datasets."""

    def load_dataset(self) -> Dataset:
        data_files = {}
        if self.cfg.train_file is not None:
            data_files["train"] = self.cfg.train_file
        if self.cfg.validation_file is not None:
            data_files["validation"] = self.cfg.validation_file
        if self.cfg.test_file is not None:
            data_files["test"] = self.cfg.test_file

        data_files = data_files if data_files else None
        if self.cfg.dataset_name == 'multi':
            # Download and load the Huggingface dataset.
            try:
                dataset_names = ['my_daily_dialog', 'my_personachat', 'my_empathetic_dialogues', 'my_wizard_of_wikipedia', 'fed']
                datasets = {}
                for dataset_name in dataset_names:
                    dataset_module = import_module(f'..datasets.{dataset_name}', self.__module__)
                    dataset = load_my_dataset(
                        dataset_module,
                        name=self.cfg.dataset_config_name,
                        cache_dir=self.cfg.cache_dir,
                        data_files=data_files,
                        history_delimeter=self.cfg.history_delimeter,
                        history_size=self.cfg.history_size,
                        script_version=f'histsz_{self.cfg.history_size}',
                        hierarchical=self.cfg.hierarchical,
                    )
                    datasets[dataset_name] = dataset
                return datasets
            except: # not a customised dataset
                return load_dataset(
                    path=self.cfg.dataset_name,
                    name=self.cfg.dataset_config_name,
                    cache_dir=self.cfg.cache_dir,
                    data_files=data_files,
                )
    
    def setup(self, stage: Optional[str] = None):
        datasets = self.load_dataset()
        for name, dataset in datasets.items():
            datasets[name] = self.process_data(dataset, stage=stage)
        self.ds = datasets

    def train_dataloader(self) -> DataLoader:
        train_loaders = []
        for name, dataset in self.ds.items():
            if name == 'fed':
                continue
            dataloader = DataLoader(
                dataset["train"],
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            train_loaders.append(dataloader)
        return train_loaders
    
    def val_dataloader(self) -> DataLoader:
        val_loaders = []
        for name, dataset in self.ds.items():
            dataloader = DataLoader(
                dataset["validation"],
                batch_size=self.batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            if name == 'fed':
                fed_loader = dataloader
                continue
            val_loaders.append(dataloader)
        val_loaders.append(fed_loader) # add fed to the last
        return val_loaders

    def test_dataloader(self) -> Optional[DataLoader]:
        test_loaders = []
        for name, dataset in self.ds.items():
            if name == 'fed':
                continue
            if "test" in dataset:
                dataloader = DataLoader(
                    dataset["test"],
                    batch_size=self.batch_size,
                    num_workers=self.cfg.num_workers,
                    collate_fn=self.collate_fn,
                    pin_memory=True,
                )
                test_loaders.append(dataloader)
        return test_loaders
