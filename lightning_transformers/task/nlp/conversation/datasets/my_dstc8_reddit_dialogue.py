import io
import json
import zipfile

import datasets
from datasets.features import Sequence
from lightning_transformers.task.nlp.conversation.datasets import dataset_base

_SPLITS = {
    "train": "https://huggingface.co/datasets/roskoN/dstc8-reddit-corpus/resolve/main/training.zip",
    "validation": "https://huggingface.co/datasets/roskoN/dstc8-reddit-corpus/resolve/main/validation_date_in_domain_in.zip",
    "test": "https://huggingface.co/datasets/roskoN/dstc8-reddit-corpus/resolve/main/validation_date_in_domain_out.zip",
    # "validation_date_out_domain_in": "https://huggingface.co/datasets/roskoN/dstc8-reddit-corpus/resolve/main/validation_date_out_domain_in.zip",
    # "validation_date_out_domain_out": "https://huggingface.co/datasets/roskoN/dstc8-reddit-corpus/resolve/main/validation_date_out_domain_out.zip",
}
_DESCRIPTION = """\
The DSTC8 dataset as provided in the original form.
The only difference is that the splits are in separate zip files.
In the orignal output it is one big archive containing all splits.
"""
_CITATION = """\
@article{lee2019multi,
  title={Multi-domain task-completion dialog challenge},
  author={Lee, S and Schulz, H and Atkinson, A and Gao, J and Suleman, K and El Asri, L and Adada, M and Huang, M and Sharma, S and Tay, W and others},
  journal={Dialog system technology challenges},
  volume={8},
  pages={9},
  year={2019}
}
"""
_LICENSE = "Like the original DSTC8 Reddit dataset, this dataset is released under the MIT license."
_HOMEPAGE = "https://github.com/microsoft/dstc8-reddit-corpus"


class DSTC8(dataset_base.DatasetBase):
    """
    The DSTC8 dataset as provided in the original form.
    The only difference is that the splits are in separate zip files.
    In the orignal output it is one big archive containing all splits.
    """

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="full", version=VERSION, description="The full dataset."
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self._features(),
            # features=datasets.Features(
            #     {
            #         "id": datasets.Value("string"),
            #         "domain": datasets.Value("string"),
            #         "task_id": datasets.Value("string"),
            #         "bot_id": datasets.Value("string"),
            #         "user_id": datasets.Value("string"),
            #         "turns": Sequence(datasets.Value("string")),
            #     }
            # ),
            citation=_CITATION,
            license=_LICENSE,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dl_paths = dl_manager.download(_SPLITS)

        return [
            datasets.SplitGenerator(
                name=split_name,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"data_path": split_path},
            )
            for split_name, split_path in dl_paths.items()
        ]

    def _generate_examples(self, data_path: str):
        with zipfile.ZipFile(data_path) as zip_file:
            dialogs = []
            for file in zip_file.namelist():
                with io.TextIOWrapper(zip_file.open(file), encoding="utf-8") as f:
                    for line in f:
                        line = json.loads(line)
                        dialogs.append(line['turns'])
            
            return self._common_generate_examples(dialogs)
