This repo is for our paper _Training a Turn-level User Engagingness Predictor for Dialogues with Weak Supervision_

This repo is based on [Lighning Transformers](https://github.com/PyTorchLightning/lightning-transformers).
More details about the changes will come later.


## Installation from source

Clone and change directory to this repo's root dir.
Then `pip install .`

## Training

`python lit.py --config-name dpp_hier_multi dataset.cfg.history_size=3 trainer.default_root_dir='your_path_to_save_checkpoints'`

## Inference

```
export DATASET=fed # or daily_dialog_engaging

python lit.py --config-name dpp_hier dataset.cfg.history_size=3 trainer.default_root_dir='your_path_to_save_checkpoints' stage=test log=False dataset=nlp/text_regression/${DATASET}
```

## License

Please observe the Apache 2.0 license that is listed in this repository. In addition, the Lightning framework is Patent Pending.
