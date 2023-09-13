from mmcv import build_from_cfg
from mmcv.utils import Registry

def build_ann_pre_processing(cfg):
    cfg_ = cfg.copy()
    build_ann_pre_processing_fn = build_from_cfg(cfg_, GTINFERENCE)
    return build_ann_pre_processing_fn

GTINFERENCE = Registry("ground_truth_inference")
