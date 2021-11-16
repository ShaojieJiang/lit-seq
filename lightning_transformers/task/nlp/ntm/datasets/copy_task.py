import os

import datasets

from lightning_transformers.task.nlp.ntm.datasets import dataset_base
import random
import numpy as np
import torch


_CITATION = """\
"""

_DESCRIPTION = """\
"""


class CopyTask(dataset_base.DatasetBase):
    """"""

    VERSION = datasets.Version("0.0.1")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self._features(),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "train",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "split": "valid",
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "split": "test",
            #     },
            # ),
        ]

    def _generate_examples(self, split):
        """Yields examples."""
        for i in range(self.num_exs):
            seq_len = random.randint(self.min_len, self.max_len)
            seq = np.random.binomial(1, 0.5, (seq_len, self.seq_width))

            inp = np.zeros((seq_len + 1, self.seq_width + 1))
            inp[:seq_len, :self.seq_width] = seq
            inp[seq_len, self.seq_width] = 1.0 # delimiter in our control channel

            yield f'{split}-{i}', {
                    "seq": inp,
                }
