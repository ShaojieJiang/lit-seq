"""TODO(fed): Add a description here."""


import json

import datasets
import numpy as np

# TODO(fed): BibTeX citation
_CITATION = """\

"""

# TODO(fed):
_DESCRIPTION = """\

"""
_URL = "http://shikib.com/fed_data.json"


class FED(datasets.GeneratorBasedBuilder):
    """TODO(fed): Short description of my dataset."""

    # TODO(fed): Set up version.
    VERSION = datasets.Version("1.0.0")

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
            for anno_id, row in enumerate(data):
                if 'response' in row: # we only want turn-level annotations
                    text = row['response']
                    engaging = row['annotations']['Engaging']
                    avg_engaging = np.mean(engaging)
                    norm10 = avg_engaging * 5

                    yield f'{anno_id}', {
                        "text": text,
                        "label": norm10,
                    }
