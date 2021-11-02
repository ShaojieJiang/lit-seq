"""TODO(empathetic_dialogues): Add a description here."""


import csv
from collections import defaultdict

import datasets
import numpy as np

from lightning_transformers.task.nlp.text_regression.datasets import dataset_base

_CITATION = """\
"""

_DESCRIPTION = """\
"""
_URL = "https://raw.githubusercontent.com/PlusLabNLP/PredictiveEngagement/master/data/DailyDialog_groungtruth_annotated.csv"


class DailyDialogEngaging(dataset_base.DatasetBase):
    """TODO(empathetic_dialogues): Short description of my dataset."""

    # TODO(empathetic_dialogues): Set up version.
    VERSION = datasets.Version("1.0.2") # norm to [0, 1]

    def _info(self):
        # TODO(empathetic_dialogues): Specifies the datasets.DatasetInfo object
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
            homepage="https://github.com/PlusLabNLP/PredictiveEngagement",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(empathetic_dialogues): Downloads the data and defines the splits
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
        # TODO(empathetic_dialogues): Yields (key, example) tuples from the dataset
        annotations = defaultdict(list)
        with open(filepath, encoding="utf-8") as f:
            data = csv.DictReader(f)
            for row in data:
                for i in range(1, 11):
                    query = row[f'Input.query{i}']
                    response = row[f'Input.response{i}']
                    anno = row[f'Answer.pair{i}']
                    annotations[f'{query}\n{response}'].append(int(anno))
        
        dialogs = []
        for key, val in annotations.items():
            query, response = key.split('\n')
            dialog = [(query, None), (response, (np.mean(val) - 1) / 4)] # normalise avg score to [0, 1]
            dialogs.append(dialog)
        
        return self._common_generate_examples(dialogs)
