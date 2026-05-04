import sys

try:
    from pytorch_lightning.callbacks import ProgressBarBase
except ImportError:
    # For newer versions of PyTorch Lightning
    from pytorch_lightning.callbacks import Callback as ProgressBarBase
from tqdm import tqdm


class EpochAndStepProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.step_bar = None
        self.epoch_bar = None
        self.refresh_rate = refresh_rate
        self.sanity_checking = False

    def on_train_start(self, trainer, pl_module):
        # Don't call super() to avoid signature mismatch issues
        self.epoch_bar = tqdm(
            desc="Training", unit="epoch", leave=True, dynamic_ncols=True, total=trainer.max_epochs
        )

        # Get total batches from trainer
        try:
            # Try new API first
            total_train = trainer.num_training_batches
            total_val_batches = trainer.num_val_batches
            if isinstance(total_val_batches, int):
                total_val = total_val_batches
            else:
                total_val = total_val_batches[0] if total_val_batches else 0
        except AttributeError:
            # Fallback to old API
            try:
                total_train = self.total_train_batches
                total_val = self.total_val_batches
            except AttributeError:
                # If both fail, use a reasonable default
                total_train = 100
                total_val = 10

        self.step_bar = tqdm(
            desc="Epoch (train)",
            unit="step",
            leave=False,
            dynamic_ncols=True,
            total=total_train + total_val,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        # Don't call super() to avoid signature mismatch issues

        self.step_bar.reset()
        self.step_bar.set_description_str("Epoch (train)")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Don't call super() to avoid signature mismatch issues

        # Get loss from callback_metrics or outputs
        try:
            loss = trainer.callback_metrics.get("loss", trainer.callback_metrics.get("train_loss", 0.0))
            if hasattr(loss, 'item'):
                loss = loss.item()
        except (KeyError, AttributeError):
            # Fallback: try to get from outputs
            if outputs is not None and isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
                if hasattr(loss, 'item'):
                    loss = loss.item()
            else:
                loss = 0.0

        self.step_bar.set_postfix_str(f"loss={loss}")
        self.step_bar.update()

    def on_validation_epoch_start(self, trainer, pl_module):
        # Don't call super() to avoid signature mismatch issues
        if self.sanity_checking:
            return

        self.step_bar.set_description_str("Epoch (validation)", True)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Don't call super() to avoid signature mismatch issues
        if self.sanity_checking:
            return

        self.step_bar.update()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Don't call super() to avoid signature mismatch issues
        if self.sanity_checking:
            return

        # NOTE(degraff): not sure why this try/except block is
        # necessary right now but for some reason the first training epoch
        # won't log the 'train_loss' key in trainer.callback_metrics
        try:
            train_loss = trainer.callback_metrics["train_loss"].item()
            val_loss = trainer.callback_metrics["val_loss"].item()
            self.epoch_bar.set_postfix_str(
                f"train_loss={train_loss:0.3f}, val_loss={val_loss:0.3f})", refresh=False
            )
        except KeyError:
            pass

        self.epoch_bar.update()

    def on_train_end(self, trainer, pl_module):
        # Don't call super() to avoid signature mismatch issues

        self.epoch_bar.close()
        self.step_bar.close()

    def on_sanity_check_start(self, trainer, pl_module):
        # Don't call super() to avoid signature mismatch issues

        self.sanity_checking = True
        print("Sanity check ...", end=" ", file=sys.stderr)

    def on_sanity_check_end(self, trainer, pl_module):
        # Don't call super() to avoid signature mismatch issues

        self.sanity_checking = False
        print("Done!", file=sys.stderr, end="\r")
