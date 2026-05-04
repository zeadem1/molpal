import logging
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.nn import functional as F

from molpal.models import mpnn
from molpal.models.chemprop.nn_utils import NoamLR

logging.getLogger("lightning").setLevel(logging.FATAL)


class LitMPNN(pl.LightningModule):
    """A message-passing neural network base class"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}

        self.mpnn = config.get("model", mpnn.MoleculeModel())
        self.uncertainty = config.get("uncertainty", "none")
        self.dataset_type = config.get("dataset_type", "regression")

        self.warmup_epochs = config.get("warmup_epochs", 2.0)
        self.max_epochs = config.get("max_epochs", 50)
        self.num_lrs = 1
        self.init_lr = config.get("init_lr", 1e-4)
        self.max_lr = config.get("max_lr", 1e-3)
        self.final_lr = config.get("final_lr", 1e-4)

        self.criterion = mpnn.utils.get_loss_func(self.dataset_type, self.uncertainty)
        self.metric = {
            "mse": lambda X, Y: F.mse_loss(X, Y, reduction="none"),
            "rmse": lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction="none")),
        }.get(config.get("metric", "rmse"), lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction="none")))

    def training_step(self, batch: Tuple, batch_idx) -> torch.Tensor:
        Xs, Y = batch

        mask = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y, nan=0.0)
        class_weights = torch.ones_like(Y)

        Y_pred = self.mpnn(Xs)
        # if args.dataset_type == 'multiclass':
        #     targets = targets.long()
        #     loss = (torch.cat([
        #         loss_func(preds[:, target_index, :],
        #                    targets[:, target_index]).unsqueeze(1)
        #         for target_index in range(preds.size(1))
        #         ], dim=1) * class_weights * mask
        #     )

        if self.uncertainty == "mve":
            Y_pred_mean = Y_pred[:, 0::2]
            Y_pred_var = Y_pred[:, 1::2]

            L = self.criterion(Y_pred_mean, Y_pred_var, Y)
        else:
            L = self.criterion(Y_pred, Y) * class_weights * mask

        loss = L.sum() / mask.sum()

        # Save loss for epoch end
        if not hasattr(self, '_train_epoch_losses'):
            self._train_epoch_losses = []
        self._train_epoch_losses.append(loss.detach())

        return loss

    def on_train_epoch_end(self):
        # Access outputs saved during training_step
        if hasattr(self, '_train_epoch_losses') and len(self._train_epoch_losses) > 0:
            train_loss = torch.stack(self._train_epoch_losses, dim=0).mean()
            self.log("train_loss", train_loss)
            self._train_epoch_losses = []

    def validation_step(self, batch: Tuple, batch_idx) -> List[float]:
        Xs, Y = batch

        Y_pred = self.mpnn(Xs)
        if self.uncertainty == "mve":
            Y_pred = Y_pred[:, 0::2]

        loss = self.metric(Y_pred, Y)

        # Save loss for epoch end
        if not hasattr(self, '_val_epoch_losses'):
            self._val_epoch_losses = []
        self._val_epoch_losses.append(loss.detach())

        return loss

    def on_validation_epoch_end(self):
        # Access outputs saved during validation_step
        if hasattr(self, '_val_epoch_losses') and len(self._val_epoch_losses) > 0:
            val_loss = torch.cat(self._val_epoch_losses).mean()
            self.log("val_loss", val_loss)
            self._val_epoch_losses = []

    def configure_optimizers(self) -> List:
        opt = Adam([{"params": self.mpnn.parameters(), "lr": self.init_lr, "weight_decay": 0}])
        sched = NoamLR(
            optimizer=opt,
            warmup_epochs=[self.warmup_epochs],
            total_epochs=[self.trainer.max_epochs] * self.num_lrs,
            steps_per_epoch=self.num_training_steps,
            init_lr=[self.init_lr],
            max_lr=[self.max_lr],
            final_lr=[self.final_lr],
        )
        scheduler = {
            "scheduler": sched,
            "interval": "step" if isinstance(sched, NoamLR) else "batch",
        }

        return [opt], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        estimated_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if estimated_steps is not None:
            return int(estimated_steps)

        if self.trainer.max_steps:
            return int(self.trainer.max_steps)

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())

        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)

        num_devices = getattr(self.trainer, "num_devices", None)
        if num_devices is None:
            num_gpus = getattr(self.trainer, "num_gpus", 0)
            num_processes = getattr(self.trainer, "num_processes", 0)
            tpu_cores = getattr(self.trainer, "tpu_cores", 0)
            num_devices = max(1, num_gpus, num_processes, tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
