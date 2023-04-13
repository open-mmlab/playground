train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1200, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1200),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05))