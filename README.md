<!-- This repo is for our paper _Training a Turn-level User Engagingness Predictor for Dialogues with Weak Supervision_ -->

This repository is based on [Lightning Transformers](https://github.com/PyTorchLightning/lightning-transformers).

# Table of contents
1. [Installation](#installation-from-source)
1. [Test or interact with our checkpoints](#run-test-or-interact-with-our-pretrained-model)

## Related papers

This repository contains the official source code for the following paper:

[1] _A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration_
<!-- WESEE -->

## Changes to the original repo
Coming soon.
Tested on the following tasks:
* Language modeling
* Conversation
<!-- * Text regression -->


## Installation from source

Clone and change directory to this repo's root dir.
Then `pip install .`

## Run test or interact with our pretrained model

The pretrained checkpoints used in paper [1] are now on Hugging Face Hub, so you can easily reproduce the results reported in our paper, or `interact` with our pretained models.

Here is the [notebook](https://colab.research.google.com/drive/1cbWX7gQfuICS4b1McqOkF2I63kYn5XWL?usp=sharing) to interact with our models on Google Colab.

For reproducing the test results on your local server, or interacting with the GPT2-small model finetuned on Wikitext-103:

`python lit --config-name lm backbone.pretrained_model_name_or_path=NeuralNotwork/gpt2-ct stage=[test | interact]`

Interacting with a language model, you can get continuations to your input prefix.

For the BlenderBot dialogue model:

`python lit --config-name dialogue_multi backbone.pretrained_model_name_or_path=NeuralNotwork/blenderbot-400M-ct stage=[test | interact]`

Interacting with a dialogue model, you can get responses to your input message.

If you don't need the W&B logging, add `log=False` to the above commands.

## Use our CT objective in your work

You can use our CT objective when **pretraining** or **finetuning** your augoregressive language models.
With CT, the resulting language models will have significantly less **repetitive** generations, even with deterministic decoding such as greedy and beam search.
It only takes severel lines of code to use CT loss, around where you calculate PyTorch's `CrossEntropyLoss`.
Here is an example:
```python
import torch

# Suppose we already have the model output logits and labels (sequences of token indices).
# For example when the batch size is 10, sequence length is 50 and vocabulary size is 1000:
logits = torch.rand(10, 50, 1000)
labels = torch.randint(0, 999, (10, 50))

# This is how you normally use cross-entropy for a language model:
from torch.nn import CrossEntropyLoss
ce_criterion = CrossEntropyLoss()
ce_loss = ce_criterion(logits.view(-1, 1000), labels.view(-1))

# This is how you can use our contrastive token loss:
from lightning_transformers.core.utils import ContrastiveTokenLoss
ct_criterion = ContrastiveTokenLoss(pad_id=999) # we need pad tokens for masking out tokens in a sequence that should not be used as negative tokens
ct_loss = ct_criterion(logits, labels)

# In our paper [1], we use CE and CT together
loss = ce_loss + ct_loss

print(ce_loss, ct_loss)

>>> tensor(6.9536) tensor(1.5848)
```

## Training

You can also reproduce our training using the instructions below.
All the data downloading and preprocessing are taken care of automatically.
All default hyper-parameters for reproducing our results are already in their corresponding `conf/*.yaml`
configuration files.
Simply run the following commands.

> **_NOTE:_**  For preprocessing big datasets such as `Wikitext-103` and `DSTC8-Reddit`, it may take longer, more CPU memory and CPU cores for the first time. But thanks to Hugging Face Datasets, once the datasets are preprocessed and cached locally, the subsequent runs should take much less memory (25GB or less) and CPU cores (usually two are enough) to run, and should be loaded instantly.

### Language modeling task

`python lit.py --config-name lm dataset.cfg.dataset_config_name=wikitext-103-raw-v1 [OPTIONS]`

For customising the training, consider the following options:
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

For customising the training, consider these options:
| optinal arguments | values | explanation |
|------|---|-------------|
| task.cfg.ct_seq_len | Positive integer | Suggested to be 1/4 (rounded) of the cross-entropy sequence length (maximum training length). Default to `30`|
| task.cfg.preced_m_negatives | Integer > -1 | `-1` means using all preceding tokens as negatives, `0` use none, k>0 uses `k`. Suggested to be 1/8 of the cross-entropy sequence length (max training length). Default to `15` |
| task.cfg.negative_method | ct, ul, nce, simctg | Which method to use for penalizing negative tokens. `ct`: contrastive token; `ul`: unlikelihood training; `nce`: noise-contrastive estimation; `simctg`: SimCTG (training objective only); Default to `ct` |
| task.cfg.ul_seq | True, False | Whether to use sequence level of UL or not. Default to `False` |
| task.cfg.simctg | True, False | Whether to use simctg loss. Default to `False` |
| training.lr | Float | Learning rate. Default to `1e-5` |
| trainer.default_root_dir | Path to your checkpoint location | Default to `${HOME}/storage/trained/lit/${task.cfg.task_name}/${backbone.pretrained_model_name_or_path}_${dataset.cfg.pretrained_dataset_name}` |

<!-- ### Text regression task (engagingness evaluator)

`python lit.py --config-name rdep_hier_multi` -->

## Test or interact

To test or interact with the models trained by yourself:

`python lit --config-name [lm | dialogue_multi] trainer.default_root_dir='your_path_to_saved_checkpoints' stage=[test | interact]`

<!-- ```
export DATASET=fed # or daily_dialog_engaging

python lit.py --config-name rdep_hier dataset.cfg.history_size=3 trainer.default_root_dir='your_path_to_save_checkpoints' stage=test log=False dataset=nlp/text_regression/${DATASET}
``` -->

## License

Please observe the Apache 2.0 license that is listed in this repository.
