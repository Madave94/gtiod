from .builder import GTINFERENCE
from .base_gt_aggregator import BaseAggregator

@GTINFERENCE.register_module()
class DisagreementGTI(BaseAggregator):

    def __init__(self, new_ann_path: str, annotator_key: str, iou_threshold: float, alpha_threshold: float):
        new_ann_path = "iou{}_alpha{}_".format(iou_threshold, alpha_threshold) + new_ann_path
        super(DisagreementGTI, self).__init__(new_ann_path, annotator_key, iou_threshold)
        self.alpha_threshold = alpha_threshold
        self.alpha = {}
        self.vitality = {}

