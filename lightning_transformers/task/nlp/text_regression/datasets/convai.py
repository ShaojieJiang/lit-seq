"""TODO(fed): Add a description here."""


import json

import datasets
import numpy as np

from lightning_transformers.task.nlp.text_regression.datasets import dataset_base

# TODO(fed): BibTeX citation
_CITATION = """\

"""

# TODO(fed):
_DESCRIPTION = """\

"""
_URL = "http://convai.io/2017/data/train_full.json"


class ConvAI(dataset_base.DatasetBase):
    """TODO(fed): Short description of my dataset."""

    # TODO(fed): Set up version.
    VERSION = datasets.Version("1.0.2") # norm to [0, 1]

    def _info(self):
        # TODO(fed): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=self._features(),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(fed): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        data_file = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_file},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_file},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_file},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(fed): Yields (key, example) tuples from the dataset
        # dialog_texts = []
        # annotations = {}
        dialogs = []
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for row in data:
                engagements = {}
                for anno in row['evaluation']:
                    engagements[anno['userId']] = anno['engagement'] / 5
                
                dialog = []
                merged_thread = []
                # merge continuous turns from same speaker
                for turn in row['thread']:
                    if merged_thread and turn['userId'] == merged_thread[-1]['userId']:
                        merged_thread[-1]['text'] = merged_thread[-1]['text'].strip() + ' ' + turn['text'].strip()
                    else:
                        merged_thread.append(turn)

                for turn in merged_thread:
                    dialog.append((turn['text'], engagements[turn['userId']]))
                
                if len(dialog) > 1:
                    dialogs.append(dialog)
        
        # yield examples
        return self._common_generate_examples(dialogs)
        