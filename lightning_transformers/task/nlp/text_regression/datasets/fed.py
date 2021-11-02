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
        dialog_texts = []
        annotations = {}
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for dialog_id, row in enumerate(data):
                if 'response' in row: # a turn-level annotation
                    history_text = row['context'] + '\n' + row['response']
                    engaging = row['annotations']['Engaging']
                    avg_engaging = np.mean(engaging)
                    norm1 = avg_engaging / 2
                    annotations[history_text] = norm1
                else: # a full-dialog
                    dialog_texts.append(row['context'])
        
        # match dialogs with turn-level annotations:
        dialogs = []
        for context, engaging in annotations.items():
            matched = False
            for dialog_text in dialog_texts:
                if context in dialog_text:
                    turns = dialog_text.replace('System: ', '').replace('User: ', '').split('\n')
                    dialog = [(turn, None) for turn in turns]
                    turn_id = len(context.split('\n')) - 1
                    dialog[turn_id] = (dialog[turn_id][0], engaging)
                    dialogs.append(dialog) # only keep the dialogs with turn-level annotations
                    matched = True
                    break
            if not matched: # context doesn't appear in full dialogs
                turns = context.replace('System: ', '').replace('User: ', '').split('\n')
                dialog = [(turn, None) for turn in turns]
                dialog[-1] = (dialog[-1][0], engaging)
                dialogs.append(dialog)
        
        # yield examples
        return self._common_generate_examples(dialogs)
        