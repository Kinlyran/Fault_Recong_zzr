import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from torch.nn import L1Loss
import sys
sys.path.insert(0,'./code')
from models import SSLHead_2Task
from losses import SwinUNETR_SSL_Loss_2Task
from data import aug_rand, rot_rand
import optimizers
import data


class SwinUnetr_trainer_2task(pl.LightningModule):
    """Pretraining on 3D Imaging with Swin-UNETR Origin ssl tasks"""

    def __init__(
        self, model_dict: dict
    ):
        super().__init__()
        self.model_dict = model_dict

        self.model = SSLHead_2Task(**model_dict)

        self.ssl_loss = SwinUNETR_SSL_Loss_2Task()
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # --------------------------
        x = batch["image"]
        batch_size = x.shape[0]
        x, rot = rot_rand(x)
        x_augment = aug_rand(x)
        
        rot_p, rec_x = self.model(x_augment)
        imgs_recon = rec_x
        imgs = x
        loss, losses_tasks = self.ssl_loss_train(rot_p, rot, imgs_recon, imgs)

        self.log("train/total_loss", loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/rot_loss", losses_tasks[0], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/recon_loss", losses_tasks[1], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # --------------------------
        val_inputs = batch["image"]
        batch_size = val_inputs.shape[0]
        x, rot = rot_rand(val_inputs)
        x_augment = aug_rand(x)
        rot_p, rec_x = self.model(x_augment)
        imgs_recon = rec_x
        imgs = x
        loss, losses_tasks = self.ssl_loss_val(rot_p, rot, imgs_recon, imgs)

        self.log("val/total_loss", loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/rot_loss", losses_tasks[0], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/recon_loss", losses_tasks[1], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        return {"val_total_loss": loss, 
                "val_rot_loss": losses_tasks[0],
                "val_recon_loss": losses_tasks[1],
                "val_number": batch_size}

    def validation_epoch_end(self, outputs):
        val_total_loss, val_rot_loss, val_recon_loss, num_items = 0., 0., 0., 0
        for output in outputs:
            val_total_loss += output["val_total_loss"].sum().item()
            val_rot_loss += output["val_rot_loss"].sum().item()
            val_recon_loss += output["val_recon_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_total_loss = torch.tensor(val_total_loss / len(outputs))
        mean_val_rot_loss = torch.tensor(val_rot_loss / len(outputs))
        mean_val_recon_loss = torch.tensor(val_recon_loss / len(outputs))
        
        self.log("val/total_loss_avg", mean_val_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/rot_loss_avg", mean_val_rot_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/recon_loss_avg", mean_val_recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.logger.log_hyperparams(
            params={
                "model": 'SwinUNETR',
                **self.model_dict,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"total_loss": mean_val_total_loss},
        )


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={'overwrite':True})
