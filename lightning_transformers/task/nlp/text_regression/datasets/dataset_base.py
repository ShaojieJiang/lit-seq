import datasets


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