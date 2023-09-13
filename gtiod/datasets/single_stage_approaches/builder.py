from mmcv import build_from_cfg
from mmcv.utils import Registry

def build_ann_prep(cfg):
    cfg_ = cfg.copy()
    ann_processing_fn = build_from_cfg(cfg_, ANNOTATION_PREPROCESSING)
    return ann_processing_fn

ANNOTATION_PREPROCESSING = Registry("ann_prep_fn")