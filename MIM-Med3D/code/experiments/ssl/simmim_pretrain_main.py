import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

import sys
sys.path.insert(0, './code')
from models import ViTSimMIM, SwinSimMIM
from torch.nn import L1Loss
import optimizers
import data


class SimMIMtrainer(pl.LightningModule):
    """Pretraining on 3D Imaging with Masked Auto Encoder"""

    def __init__(
        self, model_name: str, model_dict: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict
        if self.model_name == 'vitsimmim_base':
            self.model = ViTSimMIM(**model_dict)
        elif self.model_name == 'swinsimmim_base':
            self.model = SwinSimMIM(**model_dict)

        self.recon_loss = L1Loss()
        self.epoch_loss_values = []
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # --------------------------
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        loss = self.recon_loss(pred_pixel_values, patches[batch_range, masked_indices])

        self.log("train/l1_loss", loss, 
                batch_size=batch_size,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True)

        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train/l1_loss_avg",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        # --------------------------
        image = batch["image"]
        pred_pixel_values, patches, batch_range, masked_indices = self.model(image)
        batch_size = pred_pixel_values.shape[0]
        loss = self.recon_loss(pred_pixel_values, patches[batch_range, masked_indices])

        self.log("val/l1_loss", loss, 
                batch_size=batch_size,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True)

        return {"val_loss": loss, "val_number": batch_size}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / len(outputs))

        self.log(
            "val/l1_loss_avg", mean_val_loss, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                # "data": self.trainer.datamodule.json_path,
                # "ds_ratio": self.trainer.datamodule.downsample_ratio,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                # "benchmark": self.trainer.benchmark,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"l1_loss": mean_val_loss},
        )


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={'overwrite':True})
