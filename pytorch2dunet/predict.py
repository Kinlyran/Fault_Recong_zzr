import argparse
import logging
import os

import numpy as np
import torch

from unet.unet_model import UNet
from unet.e2unet_model import e2UNet
import cv2

def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(full_img).unsqueeze(0).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu().squeeze(0)
        mask = torch.sigmoid(output) > out_threshold
    mask = mask.squeeze(0)
    return mask.long().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_ckpts', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
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
        net = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
    elif args.model_type == 'e2UNet':
        net = e2UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)

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
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = cv2.imread(os.path.join(args.input, filename), cv2.IMREAD_UNCHANGED)
        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            filename = filename.split('.')[0]
            out_filename = f'{filename}.png'
            cv2.imwrite(os.path.join(args.output, out_filename), mask)
            logging.info(f'Mask saved to {os.path.join(args.output, out_filename)}')

            
