"""TODO(blended_skill_talk): Add a description here."""


import os

import datasets

from lightning_transformers.task.nlp.conversation.datasets import dataset_base

# TODO(blended_skill_talk): BibTeX citation
_CITATION = """\
"""

# TODO(blended_skill_talk):
_DESCRIPTION = """\
"""
_URL = "http://parl.ai/downloads/convai2/convai2_fix_723.tgz"


class ConvAI2Conversation(dataset_base.DatasetBase):
    """TODO(blended_skill_talk): Short description of my dataset."""

    # TODO(blended_skill_talk): Set up version.
    VERSION = datasets.Version("1.0.2") # norm to [0, 1]

    def _info(self):
        # TODO(blended_skill_talk): Specifies the datasets.DatasetInfo object
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
        # TODO(blended_skill_talk): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "train_both_original_no_cands.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "valid_both_original_no_cands.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "valid_both_original_no_cands.txt")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(blended_skill_talk): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = f.readlines()
            dialogs = []
            current_dialog = []
            for line in data:
                line = line.strip()
                if line.startswith('1 '): # a new dialog
                    if current_dialog:
                        dialogs.append(current_dialog)
                        current_dialog = []
                    # current_dialog.append(line[2:])
                fields = line.split('\t')
                if len(fields) == 2: # a dialog line
                    # index first space
                    ind = fields[0].index(' ')
                    current_dialog.append(fields[0][ind+1:])
                    current_dialog.append(fields[1])
            
            # add last dialog
            dialogs.append(current_dialog)
            
            return self._common_generate_examples(dialogs)
