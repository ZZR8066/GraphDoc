import torch
from torch import nn
from libs.configs.default import counter
from transformers import Trainer
from typing import Any, Dict, Union
from torch.utils.data.distributed import DistributedSampler
from libs.utils.comm import distributed, get_rank, get_world_size
from transformers.trainer import *
from .nan_detector import NanDetector

class PreTrainer(Trainer):

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if hasattr(v, "to") and hasattr(v, "device"):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs
    
    def get_train_dataloader(self):
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        if distributed():
            sampler = DistributedSampler(self.train_dataset, get_world_size(), get_rank(), True)
            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                sampler=sampler,
                num_workers=self.args.dataloader_num_workers,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                num_workers=self.args.dataloader_num_workers,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                shuffle=True,
                drop_last=self.args.dataloader_drop_last,
                pin_memory=self.args.dataloader_pin_memory
            )
        return dataloader
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = round(self._get_learning_rate(),10)
            logs["cuda_max_memory"] = int(torch.cuda.max_memory_allocated()/1024/1024)
            logs = dict(logs, **counter.dict_mean(sync=False))
            
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)