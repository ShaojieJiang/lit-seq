This repo is for our paper _Training a Turn-level User Engagingness Predictor for Dialogues with Weak Supervision_

This repo is based on [Lightning Transformers](https://github.com/PyTorchLightning/lightning-transformers).

## Changes to the original repo
Coming soon.
Tested on the following tasks:
* Language modeling
* Conversation
* Text regression


## Installation from source

Clone and change directory to this repo's root dir.
Then `pip install .`

## Training
All the data downloading and preprocessing are taken care of automatically.
All default hyper-parameters for reproducing our results are already in their corresponding `conf/*.yaml`
configuration files. 
Simply run the following commands.
Checkpoints will be released soon.

### Language modeling task

`python lit.py --config-name lm [OPTIONS]`

key options:
| optinal arguments | values | explanation |
|------|---|-------------|
| task.cfg.ct_seq_len | Positive integer | Suggested to be 1/4 (rounded) of the cross-entropy sequence length (maximum training length). Default to `150`|
| task.cfg.preced_m_negatives | Integer > -1 | `-1` means using all preceding tokens as negatives, `0` use none, k>0 uses `k`. Suggested to be 1/8 of the cross-entropy sequence length (max training length). Default to `60` |
| task.cfg.negative_method | ct, ul, nce, simctg | Which method to use for penalizing negative tokens. `ct`: contrastive token; `ul`: unlikelihood training; `nce`: noise-contrastive estimation; `simctg`: SimCTG (training objective only); Default to `ct` |
| task.cfg.ul_seq | True, False | Whether to use sequence level of UL or not. Default to `False` |
| task.cfg.simctg | True, False | Whether to use simctg loss. Default to `False` |
| training.lr | Float | Learning rate. Default to `1e-5` |
| trainer.default_root_dir | Path to your checkpoint location | Default to `${HOME}/storage/trained/lit/${task.cfg.task_name}/${backbone.pretrained_model_name_or_path}_${dataset.cfg.pretrained_dataset_name}` |

### Dialogue task

`python lit.py --config-name dialogue_multi [OPTIONS]`

key options:
| optinal arguments | values | explanation |
|------|---|-------------|
| task.cfg.ct_seq_len | Positive integer | Suggested to be 1/4 (rounded) of the cross-entropy sequence length (maximum training length). Default to `30`|
| task.cfg.preced_m_negatives | Integer > -1 | `-1` means using all preceding tokens as negatives, `0` use none, k>0 uses `k`. Suggested to be 1/8 of the cross-entropy sequence length (max training length). Default to `15` |
| task.cfg.negative_method | ct, ul, nce, simctg | Which method to use for penalizing negative tokens. `ct`: contrastive token; `ul`: unlikelihood training; `nce`: noise-contrastive estimation; `simctg`: SimCTG (training objective only); Default to `ct` |
| task.cfg.ul_seq | True, False | Whether to use sequence level of UL or not. Default to `False` |
| task.cfg.simctg | True, False | Whether to use simctg loss. Default to `False` |
| training.lr | Float | Learning rate. Default to `1e-5` |
| trainer.default_root_dir | Path to your checkpoint location | Default to `${HOME}/storage/trained/lit/${task.cfg.task_name}/${backbone.pretrained_model_name_or_path}_${dataset.cfg.pretrained_dataset_name}` |

### Text regression task (engagingness evaluator)

`python lit.py --config-name rdep_hier_multi`

## Test or interact

`python lit --config-name [lm | dialogue_multi | rdep_hier_multi] trainer.default_root_dir='your_path_to_saved_checkpoints' stage=[test | interact]`

If you don't need the log in W&B, add `log=False` to the above command.

In the `interact` mode, you can interact with the pretained models.
With a dialogue model, you can get responses to your input message; with a language model such as GPT-2, you can get continuations to your input text.

<!-- ```
export DATASET=fed # or daily_dialog_engaging

python lit.py --config-name rdep_hier dataset.cfg.history_size=3 trainer.default_root_dir='your_path_to_save_checkpoints' stage=test log=False dataset=nlp/text_regression/${DATASET}
``` -->

## License

Please observe the Apache 2.0 license that is listed in this repository.
