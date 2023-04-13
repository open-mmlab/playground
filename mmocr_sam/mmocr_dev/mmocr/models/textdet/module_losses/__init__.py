# Copyright (c) OpenMMLab. All rights reserved.
from .db_module_loss import DBModuleLoss
from .drrg_module_loss import DRRGModuleLoss
from .fce_module_loss import FCEModuleLoss
from .pan_module_loss import PANModuleLoss
from .pse_module_loss import PSEModuleLoss
from .seg_based_module_loss import SegBasedModuleLoss
from .textsnake_module_loss import TextSnakeModuleLoss

__all__ = [
    'PANModuleLoss', 'PSEModuleLoss', 'DBModuleLoss', 'TextSnakeModuleLoss',
    'FCEModuleLoss', 'DRRGModuleLoss', 'SegBasedModuleLoss'
]
