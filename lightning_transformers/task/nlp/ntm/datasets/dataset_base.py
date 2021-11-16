import datasets


class DatasetBase(datasets.GeneratorBasedBuilder):
    """A base class that integrates all common functions"""

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        self.seq_width = kwargs.pop('seq_width')
        self.min_len = kwargs.pop('min_len')
        self.max_len = kwargs.pop('max_len')
        self.num_exs = kwargs.pop('num_exs')
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
    
    def _features(self):
        return datasets.Features(
                {
                    "seq": datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
                }
            )
