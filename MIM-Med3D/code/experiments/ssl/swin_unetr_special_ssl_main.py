import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from torch.nn import L1Loss
import sys
sys.path.insert(0,'./code')
from models import SSLHead
from losses import SwinUNETR_SSL_Loss
from data import aug_rand, rot_rand
import optimizers
import data


class SwinUnetr_trainer(pl.LightningModule):
    """Pretraining on 3D Imaging with Swin-UNETR Origin ssl tasks"""

    def __init__(
        self, model_dict: dict, train_batch_size, val_batch_size
    ):
        super().__init__()
        self.model_dict = model_dict

        self.model = SSLHead(**model_dict)

        self.ssl_loss_train = SwinUNETR_SSL_Loss(batch_size=train_batch_size)
        self.ssl_loss_val = SwinUNETR_SSL_Loss(batch_size=val_batch_size)
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # --------------------------
        x = batch["image"]
        batch_size = x.shape[0]
        x1, rot1 = rot_rand(x)
        x2, rot2 = rot_rand(x)
        x1_augment = aug_rand(x1)
        x2_augment = aug_rand(x2)
        x1_augment = x1_augment
        x2_augment = x2_augment
        
        rot1_p, contrastive1_p, rec_x1 = self.model(x1_augment)
        rot2_p, contrastive2_p, rec_x2 = self.model(x2_augment)
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)
        loss, losses_tasks = self.ssl_loss_train(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

        self.log("train/total_loss", loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/rot_loss", losses_tasks[0], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/contrast_loss", losses_tasks[1], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/recon_loss", losses_tasks[2], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # --------------------------
        val_inputs = batch["image"]
        batch_size = val_inputs.shape[0]
        x1, rot1 = rot_rand(val_inputs)
        x2, rot2 = rot_rand(val_inputs)
        x1_augment = aug_rand(x1)
        x2_augment = aug_rand(x2)
        rot1_p, contrastive1_p, rec_x1 = self.model(x1_augment)
        rot2_p, contrastive2_p, rec_x2 = self.model(x2_augment)
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)
        loss, losses_tasks = self.ssl_loss_val(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

        self.log("val/total_loss", loss, batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/rot_loss", losses_tasks[0], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/contrast_loss", losses_tasks[1], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/recon_loss", losses_tasks[2], batch_size=batch_size, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        return {"val_total_loss": loss, 
                "val_rot_loss": losses_tasks[0],
                "val_contrast_loss": losses_tasks[1],
                "val_recon_loss": losses_tasks[2],
                "val_number": batch_size}

    def validation_epoch_end(self, outputs):
        val_total_loss, val_rot_loss, val_contrast_loss, val_recon_loss, num_items = 0., 0., 0., 0., 0
        for output in outputs:
            val_total_loss += output["val_total_loss"].sum().item()
            val_rot_loss += output["val_rot_loss"].sum().item()
            val_contrast_loss += output["val_contrast_loss"].sum().item()
            val_recon_loss += output["val_recon_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_total_loss = torch.tensor(val_total_loss / len(outputs))
        mean_val_rot_loss = torch.tensor(val_rot_loss / len(outputs))
        mean_val_contrast_loss = torch.tensor(val_contrast_loss / len(outputs))
        mean_val_recon_loss = torch.tensor(val_recon_loss / len(outputs))
        
        self.log("val/total_loss_avg", mean_val_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/rot_loss_avg", mean_val_rot_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/contrast_loss_avg", mean_val_contrast_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
