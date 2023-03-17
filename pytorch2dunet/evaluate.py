import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.acc import acc_score
import numpy as np


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    acc_lst = []
    dice_lst = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            mask_pred = (torch.sigmoid(mask_pred) > 0.5).long()
            # compute the acc
            acc_lst.append(acc_score(mask_pred, mask_true))
            dice_lst.append(dice_coeff(mask_pred, mask_true))
    

    net.train()
    return torch.mean(torch.tensor(acc_lst)), torch.mean(torch.tensor(dice_lst))
