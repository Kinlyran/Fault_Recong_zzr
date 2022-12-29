import argparse
import logging
import os

import numpy as np
import torch

from unet.unet_model import UNet
from unet.e2unet_model import e2UNet
import h5py

def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(full_img).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu().squeeze(0)
        mask = torch.sigmoid(output) > out_threshold
    return mask.long().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_ckpts', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--eval', '-v', action='store_true',
                        help='eval the acc')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()





if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = os.listdir(args.input)
    if not args.no_save:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    if args.model_type == 'UNet':
        net = UNet(n_channels=128, n_classes=128, bilinear=args.bilinear)
    elif args.model_type == 'e2UNet':
        net = e2UNet(n_channels=128, n_classes=128, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Model type {args.model_type}')
    logging.info(f'Loading model {args.model_ckpts}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model_ckpts, map_location=device)
    # debug when load state dict using e2UNet
    if args.model_type == 'e2UNet':
        net.train() 
    net.load_state_dict(state_dict)
    net.eval()

    logging.info('Model loaded!')
    if args.eval:
        acc = []
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        f = h5py.File(os.path.join(args.input, filename),'r') 
        img = f['raw'][:]
        true_mask = f['label'][:]
        true_mask = np.squeeze(true_mask,0)
        f.close()

        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            filename = filename.split('.')[0]
            out_filename = f'{filename}.npy'
            np.save(os.path.join(args.output, out_filename), mask)
            logging.info(f'Mask saved to {os.path.join(args.output, out_filename)}')

        if args.eval:
            cur_acc = np.mean(mask==true_mask)
            print(f'Current acc is {cur_acc}')
            acc.append(cur_acc)
    if args.eval:
        print(f'ACC is {np.mean(acc)}')
            
