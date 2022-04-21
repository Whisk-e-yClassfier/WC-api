img_norm_cfg = dict(
    mean=[0.6345, 0.5812, 0.5263], std=[0.2475, 0.2586, 0.2811], to_rgb=False)
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=27,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=27,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=27,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=7)))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.05,
                rotate_limit=10,
                p=0.5),
            dict(
                type='Cutout',
                num_holes=1,
                max_h_size=40,
                max_w_size=40,
                fill_value=0,
                p=0.5),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), mean=0, p=0.5)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize',**img_norm_cfg),
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
            dict(type='Normalize',**img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='CocoDataset',
        ann_file='dataset/train.json',
        img_prefix='dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5),
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0625,
                        scale_limit=0.05,
                        rotate_limit=10,
                        p=0.5),
                    dict(
                        type='Cutout',
                        num_holes=1,
                        max_h_size=40,
                        max_w_size=40,
                        fill_value=0,
                        p=0.5),
                    dict(
                        type='GaussNoise',
                        var_limit=(10.0, 50.0),
                        mean=0,
                        p=0.5)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(type='Normalize',**img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('Aberfeldy 12', 'Aberlour 12', 'Ardbeg 10', 'Balvenie 12',
                 'Bowmore 12', 'Bushmills', 'Chivals Regal 12',
                 'Cragganmore 12', 'Famous grouse', 'Glen Grant 10',
                 'Glenfiddich 12', 'Glenfiddich 18', 'Glenlivet 12',
                 'Highland Park 12', 'J-B', 'Jim Beam', 'Johnnie walker black',
                 'Johnnie walker red', 'Laphroaig 10', 'Macallan 12',
                 'Maker-s Mark', 'Monkey shoulder', 'Nikka coffey',
                 'Singleton 12', 'Talisker 10', 'Wild Turkey 101',
                 'Woodford Reserve'),
        data_root='/userhome/cs/u3544547/mmdetection/project/'),
    val=dict(
        type='CocoDataset',
        ann_file='dataset/test.json',
        img_prefix='dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True)
                    dict(type='Normalize',**img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Aberfeldy 12', 'Aberlour 12', 'Ardbeg 10', 'Balvenie 12',
                 'Bowmore 12', 'Bushmills', 'Chivals Regal 12',
                 'Cragganmore 12', 'Famous grouse', 'Glen Grant 10',
                 'Glenfiddich 12', 'Glenfiddich 18', 'Glenlivet 12',
                 'Highland Park 12', 'J-B', 'Jim Beam', 'Johnnie walker black',
                 'Johnnie walker red', 'Laphroaig 10', 'Macallan 12',
                 'Maker-s Mark', 'Monkey shoulder', 'Nikka coffey',
                 'Singleton 12', 'Talisker 10', 'Wild Turkey 101',
                 'Woodford Reserve'),
        data_root='/userhome/cs/u3544547/mmdetection/project/'),
    test=dict(
        type='CocoDataset',
        ann_file='dataset/test.json',
        img_prefix='dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='Normalize',**img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Aberfeldy 12', 'Aberlour 12', 'Ardbeg 10', 'Balvenie 12',
                 'Bowmore 12', 'Bushmills', 'Chivals Regal 12',
                 'Cragganmore 12', 'Famous grouse', 'Glen Grant 10',
                 'Glenfiddich 12', 'Glenfiddich 18', 'Glenlivet 12',
                 'Highland Park 12', 'J-B', 'Jim Beam', 'Johnnie walker black',
                 'Johnnie walker red', 'Laphroaig 10', 'Macallan 12',
                 'Maker-s Mark', 'Monkey shoulder', 'Nikka coffey',
                 'Singleton 12', 'Talisker 10', 'Wild Turkey 101',
                 'Woodford Reserve'),
        data_root='/userhome/cs/u3544547/mmdetection/project/'))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', classwise=True)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step', 
    gamma = 0.5,
    warmup= None, 
    step=[10, 20])
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='aiappl',
                name='cascade_rcnn_collective_ft',
                entity='smkim7'))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco/cascade_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.404_20200504_174853-b857be87.pth'
resume_from = './experiments/cascade_rcnn_collective/latest.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.05,
        rotate_limit=10,
        p=0.5),
    dict(
        type='Cutout',
        num_holes=1,
        max_h_size=40,
        max_w_size=40,
        fill_value=0,
        p=0.5),
    dict(type='GaussNoise', var_limit=(10.0, 50.0), mean=0, p=0.5)
]
classes = ('Aberfeldy 12', 'Aberlour 12', 'Ardbeg 10', 'Balvenie 12',
           'Bowmore 12', 'Bushmills', 'Chivals Regal 12', 'Cragganmore 12',
           'Famous grouse', 'Glen Grant 10', 'Glenfiddich 12',
           'Glenfiddich 18', 'Glenlivet 12', 'Highland Park 12', 'J-B',
           'Jim Beam', 'Johnnie walker black', 'Johnnie walker red',
           'Laphroaig 10', 'Macallan 12', 'Maker-s Mark', 'Monkey shoulder',
           'Nikka coffey', 'Singleton 12', 'Talisker 10', 'Wild Turkey 101',
           'Woodford Reserve')
work_dir = './experiments/cascade_rcnn_collective_ft'
seed = 1234
total_epochs = 30
gpu_ids = range(0, 1)