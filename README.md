This repo is for our paper _Training a Turn-level User Engagingness Predictor for Dialogues with Weak Supervision_

This repo is based on [Lightning Transformers](https://github.com/PyTorchLightning/lightning-transformers).

## Changes to the original repo
Coming soon.


## Installation from source

Clone and change directory to this repo's root dir.
Then `pip install .`

## Training
All the data downloading and preprocessing are taken care of automatically.
Simply run the following command to reproduce the training in our paper for the RDEP-H3 model.
Checkpoints will be released soon.

`python lit.py --config-name dpp_hier_multi dataset.cfg.history_size=3 trainer.default_root_dir='your_path_to_save_checkpoints'`

## Inference

```
export DATASET=fed # or daily_dialog_engaging

python lit.py --config-name dpp_hier dataset.cfg.history_size=3 trainer.default_root_dir='your_path_to_save_checkpoints' stage=test log=False dataset=nlp/text_regression/${DATASET}
```

## License

Please observe the Apache 2.0 license that is listed in this repository.
