# Copyright The PyTorch Lightning team.
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
import time

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import ProgressBar, convert_inf, tqdm
from pytorch_lightning.utilities import rank_zero_info


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.training_type_plugin.reduce(max_memory)
        epoch_time = trainer.training_type_plugin.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")


class ProgressBarsWithSteps(ProgressBar):
    """Modify the default progressbar to show total num_steps.
    """
    
    def __init__(self, refresh_rate: int, process_position: int):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self.step_progress_bar = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)

        if trainer.max_steps is not None:
            max_steps = trainer.max_steps
        else:
            max_steps = trainer.max_epochs * trainer.num_training_batches
        self.step_progress_bar = tqdm(
            desc='Training step',
            position=(2*(self.process_position+1)),
            total=max_steps,
            initial=trainer.global_step,
        )
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        total_batches = self.total_train_batches + self.total_val_batches
        total_batches = convert_inf(total_batches)
        if self._should_update(self.train_batch_idx, total_batches):
            self.step_progress_bar.n = self.trainer.global_step+1
            self.step_progress_bar.refresh()
