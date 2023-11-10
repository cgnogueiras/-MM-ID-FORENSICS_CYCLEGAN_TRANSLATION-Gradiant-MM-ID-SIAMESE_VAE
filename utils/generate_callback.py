import torch
import torchvision.transforms.functional as TF
from PIL import Image
import pytorch_lightning as pl
from utils.mlflow_logger import MLflowLogger

from utils.vis_fn import save_reconstructed_images


class GenerateCallback(pl.Callback):
    def __init__(
        self,
        mlflow_logger: MLflowLogger,
        x,
        y,
        z,
        a,
        input_channels=1,
        results_path=f"results",
        every_n_epochs=1,
    ):
        super().__init__()

        self.input_imgs1 = x
        self.target_imgs1 = y
        self.input_imgs2 = z
        self.target_imgs2 = a
        self.color = input_channels

        self.results_path = results_path

        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.mlflow_logger = mlflow_logger

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0:
            # Load train metrics
            learning_rate = pl_module.trainer.optimizers[0].param_groups[0]["lr"]

            train_loss = torch.stack(pl_module.training_step_outputs).mean()
            pl_module.log("training_epoch_mean", train_loss)

            self.mlflow_logger.log_metrics(
                {"train_loss": train_loss, "learning_rate": learning_rate}, step=epoch
            )

            # free up the memory
            pl_module.training_step_outputs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0:
            val_loss = torch.stack(pl_module.validation_step_outputs).mean()
            pl_module.log("validation_epoch_mean", val_loss)
            self.mlflow_logger.log_metrics({"val_loss": val_loss}, step=epoch)

            # free up the memory
            pl_module.validation_step_outputs.clear()

            save_reconstructed_images(
                pl_module,
                self.input_imgs1,
                self.target_imgs1,
                self.input_imgs2,
                self.target_imgs2,
                color=self.color,
                results_path=self.results_path,
                epoch=str(epoch),
                n=8,
            )

            self.mlflow_logger.log_artifact(self.results_path)
