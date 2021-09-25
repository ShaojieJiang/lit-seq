"""TODO(blended_skill_talk): Add a description here."""


import json
import os

import datasets

# TODO(blended_skill_talk): BibTeX citation
_CITATION = """\
@misc{smith2020evaluating,
    title={Can You Put it All Together: Evaluating Conversational Agents' Ability to Blend Skills},
    author={Eric Michael Smith and Mary Williamson and Kurt Shuster and Jason Weston and Y-Lan Boureau},
    year={2020},
    eprint={2004.08449},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

# TODO(blended_skill_talk):
_DESCRIPTION = """\
A dataset of 7k conversations explicitly designed to exhibit multiple conversation modes: displaying personality, having empathy, and demonstrating knowledge.
"""
_URL = "http://parl.ai/downloads/blended_skill_talk/blended_skill_talk.tar.gz"


class BlendedSkillTalk(datasets.GeneratorBasedBuilder):
    """TODO(blended_skill_talk): Short description of my dataset."""

    # TODO(blended_skill_talk): Set up version.
    VERSION = datasets.Version("1.0.0")
    
    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.context_delimeter = kwargs.get('delimeter', ' ')
        kwargs.pop('delimeter')
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

    def _info(self):
        # TODO(blended_skill_talk): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "response": datasets.Value("string"),
                    "dialog_id": datasets.Value("int32"),
                    "turn_id": datasets.Value("int32"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://parl.ai/projects/bst/",
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
                gen_kwargs={"filepath": os.path.join(data_dir, "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "valid.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "test.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(blended_skill_talk): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for dialog_id, row in enumerate(data):
                # personas = [row["personas"][1][0], row["personas"][1][1]]
                dialog = [turn[1] for turn in row["dialog"]]

                for turn_id, turn in enumerate(dialog):
                    if turn_id == 0:
                        continue

                    yield f'{dialog_id}-{turn_id}', {
                        "context": self.context_delimeter.join(dialog[:turn_id]),
                        "response": turn,
                        "dialog_id": dialog_id,
                        "turn_id": turn_id,
                    }
