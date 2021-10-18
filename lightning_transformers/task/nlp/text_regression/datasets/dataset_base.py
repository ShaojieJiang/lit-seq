import datasets
import random


class DatasetBase(datasets.GeneratorBasedBuilder):
    """A base class that integrates all common functions"""

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.history_delimeter = kwargs.pop('history_delimeter')
        self.history_size = kwargs.pop('history_size')
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
    
    def _common_generate_examples(self, dialogs):
        for dialog_id, dialog in enumerate(dialogs):
            dialog_len = len(dialog)
            for turn_id, turn in enumerate(dialog):
                label = dialog_len - turn_id - 1
                norm1 = label / (dialog_len - 1)

                history = dialog[:turn_id + 1]
                if self.history_size > 0:
                    history_to_keep = history[-self.history_size:]
                else:
                    history_to_keep = history
                
                history_to_keep = self._pad_random_end(history_to_keep, dialogs, dialog_id)
                history_to_keep.reverse()
                # while len(history_to_keep) < self.history_size:
                #     history_to_keep.append('') # pad empty turns

                yield f'{dialog_id}-{turn_id}', {
                    "text": self.history_delimeter.join(history_to_keep),
                    "label": norm1,
                    "dialog_id": dialog_id,
                    "turn_id": turn_id,
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