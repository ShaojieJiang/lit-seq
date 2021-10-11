"""TODO(blended_skill_talk): Add a description here."""


import os

import datasets

# TODO(blended_skill_talk): BibTeX citation
_CITATION = """\
@article{zhang2018personalizing,
  title={Personalizing dialogue agents: I have a dog, do you have pets too?},
  author={Zhang, Saizheng and Dinan, Emily and Urbanek, Jack and Szlam, Arthur and Kiela, Douwe and Weston, Jason},
  journal={arXiv preprint arXiv:1801.07243},
  year={2018}
}
"""

# TODO(blended_skill_talk):
_DESCRIPTION = """\
A dataset of 7k conversations explicitly designed to exhibit multiple conversation modes: displaying personality, having empathy, and demonstrating knowledge.
"""
_URL = "http://parl.ai/downloads/personachat/personachat.tgz"


class Personachat(datasets.GeneratorBasedBuilder):
    """TODO(blended_skill_talk): Short description of my dataset."""

    # TODO(blended_skill_talk): Set up version.
    VERSION = datasets.Version("1.0.0")
    
    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.history_delimeter = kwargs.pop('history_delimeter')
        self.history_size = kwargs.pop('history_size')
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

    def _info(self):
        # TODO(blended_skill_talk): Specifies the datasets.DatasetInfo object
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
            homepage="https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(blended_skill_talk): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        data_dir = dl_manager.download_and_extract(_URL)
        data_dir += '/personachat'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "train_both_original.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "valid_both_original.txt")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "test_both_original.txt")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(blended_skill_talk): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            dialogs = []
            current_dialog = []
            for id_, row in enumerate(f):
                splits = row.split('\t')
                if len(splits) <= 1: # persona line, skip
                    if len(current_dialog) >= 2:
                        dialogs.append(current_dialog)
                        current_dialog = []
                    continue
                context, response = splits[0], splits[1]
                context = ' '.join(context.split(' ')[1:]) # get rid of the number in the beginning
                current_dialog.extend([context, response])

        for dialog_id, dialog in enumerate(dialogs):
            dialog_len = len(dialog)
            for turn_id, turn in enumerate(dialog):
                label = dialog_len - turn_id - 1
                norm10 = label / (dialog_len - 1) * 10

                history = dialog[:turn_id + 1]
                if self.history_size > 0:
                    history_to_keep = history[-self.history_size:]
                else:
                    history_to_keep = history

                yield f'{dialog_id}-{turn_id}', {
                    "text": self.history_delimeter.join(history_to_keep),
                    "label": norm10,
                    "dialog_id": dialog_id,
                    "turn_id": turn_id,
                }
