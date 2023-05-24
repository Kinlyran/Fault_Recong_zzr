dataset_type = 'CustomDataset'
data_root = '/gpfs/share/home/2001110054/Fault_Recong/Fault_data/2Dfault_0519_256/converted'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=None,
    std=None,
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromNpy', force_3_channel=True),
    dict(type='Rerange', min_value=0.0, max_value=1.0),
    # dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=256,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.75),
    dict(type='PackInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=4,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train/image',
        with_label=False,
        pipeline=train_pipeline,
        extensions=('.npy')))
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='base',
        img_size=256,
        stage_cfgs=dict(block_cfgs=dict(window_size=7))),
    neck=dict(type='SimMIMLinearDecoder', in_channels=1024, encoder_stride=32),
    head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='PixelReconstructionLoss', criterion='L1', channel=3)))
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0008, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0),
            bias=dict(decay_mult=0.0),
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.004999999999999999,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        eta_min=4e-05,
        by_epoch=True,
        begin=10,
        end=3000,
        convert_to_iter_based=True)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3000)
auto_scale_lr = dict(base_batch_size=2048)
launcher = 'pytorch'
