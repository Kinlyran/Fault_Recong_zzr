from multi_seg_main import MultiSegtrainer
import yaml
from pytorch_lightning import Trainer
from data.Fault_dataset import FaultDataset
import os
import numpy as np
import h5py


def predict(config_path, ckpt_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(config_path,encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    model = MultiSegtrainer(**config['model']['init_args'])
    trainer = Trainer(accelerator='gpu', devices=-1, default_root_dir='./output', inference_mode=True)
    dataloader = FaultDataset(**config['data']['init_args'])
    dataloader.setup(stage='test')
    pred_loader = dataloader.test_dataloader()
    preds = trainer.predict(model, pred_loader,ckpt_path=ckpt_path)
    for i, pred in enumerate(preds):
        pred = pred[0] # only 1 class
        pred = pred.squeeze(0)
        with h5py.File(os.path.join(output_path, str(i)+'.h5'), 'w') as f:
            f['predictions'] = pred.cpu().numpy()
        
    # print(preds)
        
        


if __name__ == '__main__':
    config_path = './code/configs/sl/fault/unetr_base_vitmae_m0.75.yaml'
    ckpt_path = './output/fault/unetr_base_vitmae_p16_m0.75/checkpoints/best.ckpt'
    output_path = './output/fault/unetr_base_vitmae_p16_m0.75/preds'
    predict(config_path, ckpt_path, output_path)