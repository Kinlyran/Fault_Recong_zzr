python train.py --model_type UNet --train_dir ./data/hdf5/train --val_dir ./data/hdf5/val --ckpt_save_dir UNet_CKPTS --batch-size 4 --amp --classes 128 --epochs 20 --bilinear