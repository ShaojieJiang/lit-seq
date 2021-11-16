from importlib import import_module
from typing import Any, Callable, Dict, List, Optional

from datasets import ClassLabel, Dataset
from datasets.load import load_dataset
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorWithPadding

from lightning_transformers.core.nlp import HFDataModule
from lightning_transformers.core.utils import load_my_dataset


class NTMDataModule(HFDataModule):
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
            x for x in ["seq"] if x in dataset["train"].features
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
        return {
            'input_ids': [torch.tensor(ex) for ex in texts_or_text_pairs]
        }

    @staticmethod
    def preprocess(ds: Dataset, **fn_kwargs) -> Dataset:
        ds = ds.map(
            # todo: change this to self.convert_to_features for users to override
            NTMDataModule.convert_to_features,
            batched=True,
            with_indices=True,
            fn_kwargs=fn_kwargs,
        )
        # ds.rename_column_("seq", "input_ids")
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
            # try:
            dataset_module = import_module(f'..datasets.{self.cfg.dataset_name}', self.__module__)
            return load_my_dataset(
                dataset_module,
                name=self.cfg.dataset_config_name,
                cache_dir=self.cfg.cache_dir,
                data_files=data_files,
                # history_delimeter=self.cfg.history_delimeter,
                # history_size=self.cfg.history_size,
                script_version='',
                # hierarchical=self.cfg.hierarchical,
                seq_width=self.cfg.seq_width,
                min_len=self.cfg.min_len,
                max_len=self.cfg.max_len,
                num_exs=self.cfg.num_exs,
            )
            # except: # not a customised dataset
            #     return load_dataset(
            #         path=self.cfg.dataset_name,
            #         name=self.cfg.dataset_config_name,
            #         cache_dir=self.cfg.cache_dir,
            #         data_files=data_files,
            #     )
    
    @property
    def collate_fn(self) -> Optional[Callable]:
        return collate_fn


def collate_fn(batch):
    collated = []
    max_len = 0
    for ex in batch:
        seqs = ex['seq']
        seqs = [seq.view(1, 1, len(seq)) for seq in seqs]
        cated = torch.cat(seqs, dim=0)
        if max_len < cated.size(-1):
            max_len = cated.size(-1)
        collated.append(cated)
    
    batched = torch.zeros(collated[0].size(0), len(collated), max_len)
    for i, ex in enumerate(collated):
        curr_len = ex.size(-1)
        batched[:, i:i+1, :curr_len] = ex

    return batched
