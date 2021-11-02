"""TODO(blended_skill_talk): Add a description here."""


import json
import os

import datasets

from lightning_transformers.task.nlp.text_regression.datasets import dataset_base

# TODO(blended_skill_talk): BibTeX citation
_CITATION = """\
"""

# TODO(blended_skill_talk):
_DESCRIPTION = """\
"""
_URL = "https://parl.ai/downloads/controllable_dialogue/evaluation_logs_reproducible_v1.tar.gz"


class PersonachatEngaging(dataset_base.DatasetBase):
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
            homepage="https://parl.ai/projects/controllable_dialogue/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(blended_skill_talk): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        data_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(data_dir, 'evaluation_logs_reproducible')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "human_eval.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "human_eval.jsonl")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "human_eval.jsonl")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(blended_skill_talk): Yields (key, example) tuples from the dataset
        dialogs = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                dialog = data['dialog']
                engaging = data['evaluation_results']['enjoy']
                dialog = [(turn['text'], engaging) for turn in dialog] # using dialog-level engagingness score for each turn
                dialogs.append(dialog)

        return self._common_generate_examples(dialogs)
