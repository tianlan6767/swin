checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = ''
resume_from = '/work/Swin-Transformer-Object-Detection/work_dirs/run/train-all7-amp/epoch_280.pth'

workflow = [('train', 1)]
