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
_URL = "http://shikib.com/fed_data.json"


class FED(dataset_base.DatasetBase):
    """TODO(fed): Short description of my dataset."""

    # TODO(fed): Set up version.
    VERSION = datasets.Version("1.0.1") # norm to [0, 1]

    def _info(self):
        # TODO(fed): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.Value("float"),
                    "dialog_id": datasets.Value("int32"),
                    "turn_id": datasets.Value("int32"),
                }
            ),
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
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for dialog_id, row in enumerate(data):
                if 'response' in row: # we only want turn-level annotations
                    history_text = row['context'] + '\n' + row['response']
                    history_text = history_text.replace('System: ', '').replace('User: ', '')
                    engaging = row['annotations']['Engaging']
                    avg_engaging = np.mean(engaging)
                    norm1 = avg_engaging / 2

                    history = history_text.split('\n')
                    if self.history_size > 0:
                        history_to_keep = history[-self.history_size:]
                    else:
                        history_to_keep = history

                    yield f'{dialog_id}', {
                        "text": self.history_delimeter.join(history_to_keep),
                        "label": norm1,
                        "dialog_id": dialog_id,
                        "turn_id": len(history) - 1,
                    }
