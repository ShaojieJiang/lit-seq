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
from typing import Callable, Optional, Tuple

from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollatorForSeq2Seq

from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataModule
from lightning_transformers.core.utils import load_dataset_builder, load_my_dataset


class ConversationDataModule(Seq2SeqDataModule):
    """Defines the ``LightningDataModule`` for Conversation Datasets."""

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
            # try:
            dataset_module = import_module(f'..datasets.{self.cfg.dataset_name}', self.__module__)
            return load_my_dataset(
                dataset_module,
                name=self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                data_files=data_files,
                history_delimiter=self.cfg.history_delimiter,
                history_size=self.cfg.history_size,
                script_version=f'histsz_{self.cfg.history_size}',
                # hierarchical=self.cfg.hierarchical,
            )
            # except: # not a customised dataset
            #     return load_dataset_builder(
            #         path=self.cfg.dataset_name,
            #         name=self.cfg.dataset_config_name,
            #         cache_dir=self.cfg.cache_dir,
            #         data_files=data_files,
            #     )
        
    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return "context", "response"
    
    @property
    def collate_fn(self) -> Optional[Callable]:
        if self.cfg.padding != 'max_length':
            return DataCollatorForSeq2Seq(self.tokenizer)
        else:
            return super().collate_fn


class ConversationMultiDataModule(ConversationDataModule):

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
            # try:
            dataset_names = self.cfg.dataset_components.split(':')
            datasets = {}
            for dataset_name in dataset_names:
                dataset_module = import_module(f'..datasets.{dataset_name}', self.__module__)
                dataset = load_my_dataset(
                    dataset_module,
                    name=self.cfg.dataset_config_name,
                    cache_dir=self.cfg.cache_dir,
                    data_files=data_files,
                    history_delimiter=self.cfg.history_delimiter,
                    history_size=self.cfg.history_size,
                    script_version=f'histsz_{self.cfg.history_size}',
                    # hierarchical=self.cfg.hierarchical,
                )
                datasets[dataset_name] = dataset
            return datasets
    
    def setup(self, stage: Optional[str] = None):
        datasets = self.load_dataset()
        for name, dataset in datasets.items():
            datasets[name] = self.process_data(dataset, stage=stage)
        self.ds = datasets

    def train_dataloader(self) -> DataLoader:
        train_loaders = []
        for name, dataset in self.ds.items():
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
                batch_size=self.eval_batch_size,
                num_workers=self.cfg.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )
            val_loaders.append(dataloader)
        return val_loaders

    def test_dataloader(self) -> Optional[DataLoader]:
        test_loaders = []
        for name, dataset in self.ds.items():
            if "test" in dataset:
                dataloader = DataLoader(
                    dataset["test"],
                    batch_size=self.eval_batch_size,
                    num_workers=self.cfg.num_workers,
                    collate_fn=self.collate_fn,
                    pin_memory=True,
                )
                test_loaders.append(dataloader)
        return test_loaders
