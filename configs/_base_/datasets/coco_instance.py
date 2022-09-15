from pickle import NONE


dataset_type = 'CocoDataset'
data_root = '/work/data/855G3/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1600, 1600)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),

    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        allow_negative_crop=True),
    
    # dict(
    #     type='Resize',
    #     img_scale=image_size,
    #     # ratio_range=(0.1, 2.0),
    #     # multiscale_mode='range',
    #     keep_ratio=True),

    # dict(
    #     type='RandomCenterCropPad',
    #     crop_size=image_size,
    #     ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
    #     test_mode=False,
    #     test_pad_mode=None,
    #     **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor = 1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val_train/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/train.json',
        # img_prefix=data_root + 'train/',
        pipeline=test_pipeline),
    )
evaluation = dict(interval=2,metric=['bbox', 'segm'])
