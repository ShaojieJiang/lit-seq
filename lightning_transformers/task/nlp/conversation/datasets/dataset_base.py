import datasets


class DatasetBase(datasets.GeneratorBasedBuilder):
    """A base class that integrates all common functions"""

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.history_delimeter = kwargs.pop('history_delimeter')
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
                    "context": text_feature,
                    "response": datasets.Value("string"),
                    "dialog_id": datasets.Value("int32"),
                    "turn_id": datasets.Value("int32"),
                }
            )
    
    def _common_generate_examples(self, dialogs):
        for dialog_id, dialog in enumerate(dialogs):
            for turn_id, turn in enumerate(dialog):
                if turn_id == 0:
                    continue # at least two turns are needed for each data point

                history = dialog[:turn_id]
                response = turn


                if self.history_size > 0:
                    history_to_keep = history[-self.history_size:]
                else:
                    history_to_keep = history
                    
                history_to_keep.reverse()
                    
                if self.hierarchical:
                    context = history_to_keep
                else:
                    context = self.history_delimeter.join(history_to_keep)

                yield f'{dialog_id}-{turn_id}', {
                    "context": context,
                    "response": response,
                    "dialog_id": dialog_id,
                    "turn_id": turn_id,
                }
