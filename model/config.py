_base_ = './configs/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco.py'
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[0.6347, 0.5802, 0.5233], std=[0.2470, 0.2582, 0.2813], to_rgb=False)
albu_train_transforms = [
            # dict(
            #     type='OneOf',
            #     transforms=[
            #         dict(
            #             type='RandomBrightnessContrast',
            #             brightness_limit=0.1,
            #             contrast_limit=0.1,
            #             p=1.0),
            #         dict(
            #             type='ColorJitter',
            #             brightness=0.1,
            #             contrast=0.1,
            #             saturation=0.1,
            #             hue=0.1,
            #             p=1.0),
            #         dict(type='CLAHE', p=1.0)
            #     ],
            #     p=0.5),
            # dict(
            #     type='OneOf',
            #     transforms=[
            #         dict(
            #             type='ShiftScaleRotate',
            #             shift_limit=0.0625,
            #             scale_limit=0.05,
            #             rotate_limit=10,
            #             p=0.4),
            #         dict(type='Affine', shear=(-10.0, 10.0), p=0.5)
            #     ],
            #     p=1.0),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.05,
                rotate_limit=10,
                p=0.4),
            dict(
                type='Cutout',
                num_holes=1,
                max_h_size=40,
                max_w_size=40,
                fill_value=0,
                p=0.4),
            # dict(
            #     type='OneOf',
            #     transforms=[
            #         dict(
            #             type='GaussNoise',
            #             var_limit=(10.0, 50.0),
            #             mean=0,
            #             p=1.0),
            #         dict(
            #             type='ISONoise',
            #             color_shift=(0.01, 0.05),
            #             intensity=(0.1, 0.5),
            #             p=1.0)
            #     ],
            #     p=0.5)
        ]
classes = ('Aberfeldy 12', 'Aberlour 12', 'Ardbeg 10', 'Balvenie 12', 'Bowmore 12', 'Bushmills', 'Chivals Regal 12', 'Cragganmore 12',
 'Famous grouse', 'Glen Grant 10', 'Glenfiddich 12', 'Glenfiddich 18', 'Glenlivet 12', 'Highland Park 12', 'J-B', 'Jim Beam', 'Johnnie walker black', 'Johnnie walker red',
 'Laphroaig 10', 'Macallan 12', 'Maker-s Mark', 'Monkey shoulder', 'Nikka coffey', 'Singleton 12', 'Talisker 10', 'Wild Turkey 101', 'Woodford Reserve')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type='CocoDataset',
        ann_file='dataset/train.json',
        img_prefix='dataset/',
        pipeline=train_pipeline,
        classes=classes,
        data_root='/userhome/cs/u3544547/mmdetection/project/'),
    val=dict(
        type='CocoDataset',
        ann_file='dataset/test.json',
        img_prefix='dataset/',
        pipeline=test_pipeline,
        classes=classes,
        data_root='/userhome/cs/u3544547/mmdetection/project/'),
    test=dict(
        type='CocoDataset',
        ann_file='dataset/test.json',
        img_prefix='dataset/',
        pipeline=test_pipeline,
        classes=classes,
        data_root='/userhome/cs/u3544547/mmdetection/project/'))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', classwise=True)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=2.0, norm_type=2))
lr_config = dict(policy='step', gamma=0.5, warmup=None, step=[15, 25])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='aiappl',
                name='exp-v3_fcos_hrnet-384',
                entity='smkim7'))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752-f22d2ce5.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w40')),
    neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=27,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.8,
            neg_iou_thr=0.5,
            min_pos_iou=0.8,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
        pos_iou_thr=0.8),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.1,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=5))


work_dir = './experiments/v3_fcos_hrnet-384'
seed = 1234
total_epochs = 10
gpu_ids = range(0, 1)

