norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=0,
    size=(512, 512))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MAE',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=[3, 5, 7, 11],
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=backbone_norm_cfg,
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=1.0,
        init_cfg=dict(type='Pretrained',checkpoint='/gpfs/share/home/2001110054/Fault_Recong/mmpretrain/output/mae_vit-base-p16_8xb512-amp-coslr-300e_mix_force_3_chan_per_image_norm/epoch_300.pth')),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        out_channels=1,
        num_classes=2,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        out_channels=1,
        num_classes=2,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
dataset_type = 'FaultDataset'
project_data_root_lst = ['../Fault_data/real_labeled_data/2d_slices', '../Fault_data/project_data_v1/labeled/qyb/2d_slices_sl', '../Fault_data/project_data_v1/labeled/Ordos/gjb/2d_slices_sl', '../Fault_data/project_data_v1/labeled/Ordos/pl/2d_slices_sl', '../Fault_data/project_data_v1/labeled/Ordos/yw/2d_slices_sl']
thebe_data_root = '../Fault_data/public_data/2d_slices'
crop_size = (512, 512)
thebe_train_pipeline = [
    dict(type='LoadImageFromNpy', force_3_channel=True),
    dict(type='PerImageNormalization', ignore_zoro=True),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.99),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
project_data_train_pipline = [
    dict(type='LoadImageFromNpy', force_3_channel=True),
    dict(type='PerImageNormalization', ignore_zoro=True),
    dict(type='LoadAnnotations', dilate=True),
    # dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.99),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromNpy', force_3_channel=True),
    dict(type='PerImageNormalization', ignore_zoro=True),
    # dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataset_lst = []
thebe_train_dataset = dict(type=dataset_type,
                            data_root=thebe_data_root,
                            data_prefix=dict(
                            img_path='train/image', seg_map_path='train/ann'),
                            pipeline=thebe_train_pipeline)
train_dataset_lst.append(thebe_train_dataset)
for data_root in project_data_root_lst:
    train_dataset_lst.append(dict(type=dataset_type,
                            data_root=data_root,
                            data_prefix=dict(
                            img_path='train/image', seg_map_path='train/ann'),
                            pipeline=project_data_train_pipline))

train_dataset = dict(type='ConcatDataset',
                     datasets=train_dataset_lst)
thebe_val_dataset = dict(type=dataset_type,
                            data_root=thebe_data_root,
                            data_prefix=dict(
                            img_path='val/image',
                            seg_map_path='val/ann'),
                            pipeline=test_pipeline)
"""
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

"""
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=thebe_val_dataset)
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mFscore'])
test_evaluator = val_evaluator
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
# tta_model = dict(type='SegTTAModel')
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=1600)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1600),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
fp16 = dict(loss_scale='dynamic')
launcher = 'pytorch'
