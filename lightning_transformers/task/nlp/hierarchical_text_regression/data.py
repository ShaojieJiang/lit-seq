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

import torch
from datasets import ClassLabel, Dataset
from datasets.load import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorWithPadding

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.core.utils import load_my_dataset


class HierarchicalTextRegressionDataModule(HFDataModule):
    """Defines the ``LightningDataModule`` for Text Regression Datasets."""

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        input_feature_fields = [k for k, v in dataset["train"].features.items() if k not in ["label", "idx"]]
        dataset = dataset.sort('sort_key', reverse=True) # should CUDA OOM exist, this allows it to appear earlier
        dataset = HierarchicalTextRegressionDataModule.preprocess(
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
    
        def tokenize_list_texts(text_list):
            tokenized = []
            for text in text_list:
                tokenized.append(tokenizer(text, **tokenizer_kwargs))
            tokenized_dict = {}
            keys = list(tokenized[0].keys())
            for key in keys:
                tokenized_dict[key] = [res[key] for res in tokenized]
            return tokenized_dict

        text_lists = example_batch[input_feature_fields[0]]
        # tokenize lists of texts
        tokenized = list(map(tokenize_list_texts, text_lists))
        tokenized_dict = {}
        keys = list(tokenized[0].keys())
        for key in keys:
            tokenized_dict[key] = [res[key] for res in tokenized]


        return tokenized_dict

    @staticmethod
    def preprocess(ds: Dataset, **fn_kwargs) -> Dataset:
        ds = ds.map(
            # todo: change this to self.convert_to_features for users to override
            HierarchicalTextRegressionDataModule.convert_to_features,
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
                    history_size=self.cfg.history_size,
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
            return DataCollatorWithTurnAndDialogPadding(self.tokenizer)
        else:
            return super().collate_fn


class DataCollatorWithTurnAndDialogPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # pad dialogues to the same length
        max_turns = max(len(feature['input_ids']) for feature in features)
        self.pad_dialogues(features, max_turns)
        # pad i-th turn of each dialogue to the same length
        keys = ['attention_mask', 'input_ids', 'token_type_ids']
        turn_batches = []
        for i in range(max_turns):
            batch_turn_i = {key: [feature[key][i] for feature in features] for key in keys}
            batch_turn_i = self.tokenizer.pad(
                batch_turn_i,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            turn_batches.append(batch_turn_i)
        
        batch = {
            'turn_batches': turn_batches,
        }
        other_features = set(features[0].keys()) - set(keys)
        other_feature_batch = {key: torch.tensor([feature[key] for feature in features]) for key in other_features}
        
        batch.update(other_feature_batch)
        
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
    
    def pad_dialogues(self, batch_dialogues, max_turns):
        keys = ['attention_mask', 'input_ids', 'token_type_ids']
        dummy_turn = {'input_ids': torch.LongTensor([101, 102]), 'attention_mask': torch.LongTensor([1, 1]), 'token_type_ids': torch.LongTensor([0, 0])}
        for dialogue in batch_dialogues:
            if len(dialogue[keys[0]]) < max_turns: # needs to be padded
                pad_len = max_turns - len(dialogue[keys[0]])
                for key in keys:
                    dialogue[key] += pad_len * [dummy_turn[key]]
