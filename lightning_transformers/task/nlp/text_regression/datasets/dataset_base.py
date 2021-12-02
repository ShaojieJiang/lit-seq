import datasets
import random


class DatasetBase(datasets.GeneratorBasedBuilder):
    """A base class that integrates all common functions"""

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.history_delimiter = kwargs.pop('history_delimiter')
        self.history_size = kwargs.pop('history_size')
        self.hierarchical = kwargs.pop('hierarchical', False)
        if self.hierarchical:
            hier_version_str = f'{self.VERSION.major}.{self.VERSION.minor}.{self.VERSION.patch + 100}'
            self.VERSION = datasets.Version(hier_version_str)
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
    
    def _features(self):
        if self.hierarchical:
            text_feature = datasets.Sequence(datasets.Value("string"))
        else:
            text_feature = datasets.Value("string")
        return datasets.Features(
                {
                    "text": text_feature,
                    "label": datasets.Value("float"),
                    "dialog_id": datasets.Value("int32"),
                    "turn_id": datasets.Value("int32"),
                    # "sort_key": datasets.Value("int32"),
                }
            )
    
    def _common_generate_examples(self, dialogs):
        for dialog_id, dialog in enumerate(dialogs):
            dialog_len = len(dialog)
            for turn_id, turn in enumerate(dialog):
                if type(turn) is str:
                    label = dialog_len - turn_id - 1
                    norm1 = label / (dialog_len - 1)

                    history = dialog[:turn_id + 1]
                elif type(turn) is tuple: # a turn with human annotations
                    _, engaging = turn
                    if engaging is None: # not every turn is annotated
                        continue

                    norm1 = engaging
                    history = [turn[0] for turn in dialog[:turn_id + 1]]

                if self.history_size > 0:
                    history_to_keep = history[-self.history_size:]
                else:
                    history_to_keep = history
                    
                # history_to_keep = self._pad_random_end(history_to_keep, dialogs, dialog_id)
                history_to_keep.reverse()
                # while len(history_to_keep) < self.history_size:
                #     history_to_keep.append('') # pad empty turns
                    
                if self.hierarchical:
                    text = history_to_keep
                else:
                    text = self.history_delimiter.join(history_to_keep)

                # if len(history_to_keep) > 1 and type(turn) is str: # example w/o history, not for human data
                #     last_turn = history_to_keep[0]
                #     if self.hierarchical:
                #         yield f'{dialog_id}-{turn_id}-{0}', {
                #             "text": [last_turn],
                #             "label": norm1,
                #             "dialog_id": dialog_id,
                #             "turn_id": turn_id,
                #             # "sort_key": len(last_turn.split()),
                #         }
                #     else:
                #         yield f'{dialog_id}-{turn_id}-{0}', {
                #             "text": last_turn,
                #             "label": norm1,
                #             "dialog_id": dialog_id,
                #             "turn_id": turn_id,
                #             # "sort_key": len(last_turn.split()),
                #         }

                # example w/ history
                yield f'{dialog_id}-{turn_id}', {
                    "text": text,
                    "label": norm1,
                    "dialog_id": dialog_id,
                    "turn_id": turn_id,
                    # "sort_key": len(text.split()),
                }

    def _pad_random_end(self, history, dialogs, dialog_id):
        turns_needed = self.history_size - len(history)
        if turns_needed <= 0:
            return history # no padding needed

        rand_dialog_id = None
        while True:
            rand_dialog_id = random.randrange(len(dialogs))
            if rand_dialog_id == dialog_id or len(dialogs[rand_dialog_id]) < turns_needed:
                continue
            break
        # found rand_dialog_id
        pads = dialogs[rand_dialog_id][-turns_needed:]
        history = pads + history

        return history
