# 基于Transformer的断层识别

# 代码库简介

 - 全部使用pytorch深度学习框架
 - 2D模型基于[mmcv](https://github.com/open-mmlab/mmcv), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 以及[mmpretrain](https://github.com/open-mmlab/mmpretrain).
 - 3D模型基于[pytorch-lightning](https://github.com/Lightning-AI/lightning)开发
 - 2D分割模型代码: [mmsegmentation](./mmsegmentation/), 项目相关配置文件位于[mmsegmentation/projects/Fault_recong](./mmsegmentation/projects/Fault_recong)
 - 2D预训练代码: [mmpretrain](./mmpretrain/), 项目相关配置文件位于[mmpretrain/projects/Fault_Recong](./mmpretrain/projects/Fault_Recong)
 - [3D分割, 预训练代码](./MIM-Med3D/), 项目相关配置文件位于[./MIM-Med3D/code/configs](./MIM-Med3D/code/configs)

# 环境安装
2D分割, 预训练, 3D分割、预训练代码相互独立, 如只需使用2D模型, 仅需要安装2D模型需要的环境即可. 由于整个项目是由pytorch开发, 首先需要安装pytorch, 我使用的是PyTorch: 1.12.1+cu113版本的torch, 由于CUDA版本不同, 可能需要安装的torch略有差异, 可直接去[官网](https://pytorch.org/get-started/pytorch-2.0/)安装torch. 推荐安装torch-1.12.1版本. **在安装完torch之后, 才可以进行2D或者3D模型的环境配置**

## 2D分割模型环境安装
进入[2D分割代码库](./mmsegmentation), 原本代码库的说明文档位于[./mmsegmentation/README_zh-CN.md](./mmsegmentation/README_zh-CN.md), 这里简单说明一下安装步骤, 如遇问题可参考mmsegmentation的官方文档.
 
步骤0: 使用MIM安装MMCV
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
如果无法使用MIM安装, 可去[MMCV官网](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html), 选择合适的torch和cuda版本, 使用pip安装. 例如安装基于cuda11.3, torch1.12.x的mmcv命令为
```
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```

步骤1: 安装MMSegmentation
```
cd mmsegmentation
pip install -v -e .
# '-v' 表示详细模式，更多的输出
# '-e' 表示以可编辑模式安装工程，
# 因此对代码所做的任何修改都生效，无需重新安装
```

## 3D分割模型环境安装
进入[3D分割代码库](./MIM-Med3D):
```
pip install -r requirements.txt
```

# 模型预测接口

## 2D模型预测接口
在[2D分割文件夹下](./mmsegmentation)

通用格式, 调用[./mmsegmentation/projects/Fault_recong/predict.py](./mmsegmentation/projects/Fault_recong/predict.py)中的predict_3d或predict_2d函数. 

其中predict_3d函数接受的输入为.npy或者.sgy文件

predict_2d函数接受的输入为包含所有需要预测的2d图片的文件夹, 里面的文件为.npy或者.png的单通道图片. 
```
python ./projects/Fault_recong/predict.py --config {Path to model config} \
                                        --checkpoint {Model checkpoint path} \
                                        --input {Input image/cube path} \
                                        --save_path {Path to save predict result} \
                                        --predict_type {Predict 2d/3d fault} \
                                        --device {Set cuda device} \
```


调用基于thebe数据训练的网络, 使用了2.5d数据拼接方式, 随机剪裁512x512分辨率(输入切片分辨率不能低于此分辨率), 断层损失函数权重10倍加权等技巧. 由于2.5d数据技巧，该模型只能用于预测3d断层. 输出是与输入相同大小的预测结果(predict.npy)和得分结果(score.npy).
```
python ./projects/Fault_recong/predict.py --config ./output/swin-base-thebe-512x512/swin-base-thebe-512x512.py \
                                        --checkpoint ./output/swin-base-thebe-512x512/Best.pth \
                                        --input /Fault_data/public_data/precessed/test/seis/seistest.npy \
                                        --save_path ./output/swin-base-thebe-512x512/predict \
                                        --predict_type 3d \
                                        --device cuda:0 \
```

调用基于项目3d数据(501x501x801大小)训练的2D网络对3D数据进行预测，由于2.5d数据技巧，该模型只能用于预测3d断层. 训练时的随机剪裁256x256, 输入的图片不能低于此分辨率. 输出是与输入相同大小的预测结果(predict.npy)和得分结果(score.npy).
```
python ./projects/Fault_recong/predict.py --config ./output/swin-base-3d_project_data-256x256/swin-base-3d_project_data-256x256.py \
                                        --checkpoint ./output/swin-base-3d_project_data-256x256/Best.pth \
                                        --input /Fault_data/real_labeled_data/origin_data/seis/mig_fill.sgy \
                                        --save_path ./output/swin-base-3d_project_data-256x256/predict \
                                        --predict_type 3d \
                                        --device cuda:0 \
```

调用基于项目2d数据(分辨率256x256, 一共991张图片)训练的2D网络对2D数据进行预测, 注意输入的图片是将原始的单通道复制三次, 形成3通道图片, predict_2d函数中的force_3_chan=True即可. 输出在save_path中，是每张图片预测的score...
```
# random split 训练出的模型
python ./projects/Fault_recong/predict.py --config ./output/swin-base-2D_0519_Data-256x256-random-split/swin-base-2D_0519_Data-256x256-random-split.py \
                                        --checkpoint ./output/swin-base-2D_0519_Data-256x256-random-split/Best.pth \
                                        --input /Fault_data/2Dfault_0519_256/converted/val/image \
                                        --save_path ./output/swin-base-2D_0519_Data-256x256-random-split/predict \
                                        --predict_type 2d \
                                        --device cuda:0 \

# 使用前三个slice训练出的模型
python ./projects/Fault_recong/predict.py --config ./output/swin-base-2D_0519_Data-256x256-slice-split/swin-base-2D_0519_Data-256x256-slice-split.py \
                                        --checkpoint ./output/swin-base-2D_0519_Data-256x256-slice-split/Best.pth \
                                        --input /Fault_data/2Dfault_0519_256/converted_slice_split/val/image \
                                        --save_path ./output/swin-base-2D_0519_Data-256x256-slice-split/predict \
                                        --predict_type 2d \
                                        --device cuda:0 \

```

## 3D模型预测接口
在[3D模型代码库](./MIM-Med3D/)下, 调用[./mmsegmentation/projects/Fault_recong/predict.py](./mmsegmentation/projects/Fault_recong/predict.py)中的predict_sliding_window函数, 模型会按照128x128x128的大小对输入的3D断层进行slice inferrence. 调用的通用格式如下
```
python ./code/experiments/sl/prediect.py --config {Path to model config} \
                                        --checkpoint {Model checkpoint path} \
                                        --input {Input image/cube path} \
                                        --save_path {Path to save predict result} \
                                        --device {Set cuda device} \
```
预测的结果以及每个像素点的得分会保存在save_path文件夹下..

调用基于Thebe数据训练的3D分割模型
```
python ./code/experiments/sl/prediect.py \
        --config ./output/swin_unetr_base_supbaseline_p16_public_192x384x384_zoom/config.yaml \
        --checkpoint ./output/swin_unetr_base_supbaseline_p16_public_192x384x384_zoom/checkpoints/best.ckpt \
        --input /Fault_data/public_data/precessed/test/seis/seistest.npy \
        --save_path ./output/swin_unetr_base_supbaseline_p16_public_192x384x384_zoom/predict \
        --device cuda:0 \
```

调用基于项目3d数据(501x501x801大小)训练的3D分割模型
```
python ./code/experiments/sl/prediect.py \
        --config ./output/swin_unetr_base_simmim_p16_real_labeled_crop_192-pos-weight-10-dilate-1/config.yaml \
        --checkpoint ./output/swin_unetr_base_simmim_p16_real_labeled_crop_192-pos-weight-10-dilate-1/checkpoints/best.ckpt \
        --input /Fault_data/real_labeled_data/origin_data/seis/mig_fill.sgy \
        --save_path  ./output/swin_unetr_base_simmim_p16_real_labeled_crop_192-pos-weight-10-dilate-1/predict \
        --device cuda:0 \
```
