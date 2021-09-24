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
from typing import Tuple

from datasets import Dataset

from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataModule
from lightning_transformers.core.utils import load_my_dataset


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
            dataset_module = import_module(f'..datasets.{self.cfg.dataset_name}', self.__module__)
            return load_my_dataset(
                dataset_module,
                name=self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                data_files=data_files,
                delimeter=self.cfg.history_delimeter,
            )
        
    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return "context", "response"
