# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and Shaojie Jiang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset"""


import os
from zipfile import ZipFile

import datasets

_CITATION = """\
@InProceedings{li2017dailydialog,
    author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
    title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
    booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
    year = {2017}
}
"""

_DESCRIPTION = """\
We develop a high-quality multi-turn dialog dataset, DailyDialog, which is intriguing in several aspects.
The language is human-written and less noisy. The dialogues in the dataset reflect our daily communication way
and cover various topics about our daily life. We also manually label the developed dataset with communication
intention and emotion information. Then, we evaluate existing approaches on DailyDialog dataset and hope it
benefit the research field of dialog systems.
"""

_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"

act_label = {
    "0": "__dummy__",  # Added to be compatible out-of-the-box with datasets.ClassLabel
    "1": "inform",
    "2": "question",
    "3": "directive",
    "4": "commissive",
}

emotion_label = {
    "0": "no emotion",
    "1": "anger",
    "2": "disgust",
    "3": "fear",
    "4": "happiness",
    "5": "sadness",
    "6": "surprise",
}


class DailyDialog(datasets.GeneratorBasedBuilder):
    """DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset"""

    VERSION = datasets.Version("1.0.0")

    __EOU__ = " __eou__ " # get rid of white spaces between turns
    
    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.context_delimeter = kwargs.get('delimeter', ' ')
        kwargs.pop('delimeter')
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "response": datasets.Value("string"),
                    "act": datasets.ClassLabel(names=list(act_label.values())),
                    "emotion": datasets.ClassLabel(names=list(emotion_label.values())),
                    "dialog_id": datasets.Value("int32"),
                    "turn_id": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
            homepage="http://yanran.li/dailydialog",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "ijcnlp_dailydialog")

        # The splits are nested inside the zip
        for name in ("train", "validation", "test"):
            zip_fpath = os.path.join(data_dir, f"{name}.zip")
            with ZipFile(zip_fpath) as zip_file:
                zip_file.extractall(path=data_dir)
                zip_file.close()

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "train", "dialogues_train.txt"),
                    "act_path": os.path.join(data_dir, "train", "dialogues_act_train.txt"),
                    "emotion_path": os.path.join(data_dir, "train", "dialogues_emotion_train.txt"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "test", "dialogues_test.txt"),
                    "act_path": os.path.join(data_dir, "test", "dialogues_act_test.txt"),
                    "emotion_path": os.path.join(data_dir, "test", "dialogues_emotion_test.txt"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "validation", "dialogues_validation.txt"),
                    "act_path": os.path.join(data_dir, "validation", "dialogues_act_validation.txt"),
                    "emotion_path": os.path.join(data_dir, "validation", "dialogues_emotion_validation.txt"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, file_path, act_path, emotion_path, split):
        """Yields examples."""
        # Yields (key, example) tuples from the dataset
        with open(file_path, "r", encoding="utf-8") as f, open(act_path, "r", encoding="utf-8") as act, open(
            emotion_path, "r", encoding="utf-8"
        ) as emotion:
            for dialog_id, (line_f, line_act, line_emotion) in enumerate(zip(f, act, emotion)):
                if len(line_f.strip()) == 0:
                    break
                dialog = line_f.replace(' __eou__\n', '').split(self.__EOU__) # get rid of the last __eou__ before splitting
                act = line_act.split(" ")[:-1]
                emotion = line_emotion.split(" ")[:-1]

                assert len(dialog) == len(act) == len(emotion), "Different turns btw dialogue & emotion & action"

                for turn_id, (turn, turn_act, turn_emotion) in enumerate(zip(dialog, act, emotion)):
                    if turn_id == 0:
                        continue # one turn only cannot form a pair

                    yield f"{split}-{dialog_id}-{turn_id}", {
                        "context": self.context_delimeter.join(dialog[:turn_id]),
                        "response": turn,
                        "act": int(turn_act),
                        "emotion": int(turn_emotion),
                        "dialog_id": dialog_id,
                        "turn_id": turn_id,
                    }