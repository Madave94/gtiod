from os import path
from datetime import datetime

from .builder import GTINFERENCE
from .weighted_boxes_fusion import WeightedBoxesFusion
from .localization_aware_em import LocalizationAwareEM

@GTINFERENCE.register_module()
class ExpectationMaximizationWBF(WeightedBoxesFusion):

    def __init__(self, new_ann_path, annotator_key, iou_threshold, confidence_threshold, merging_ops,
                 scores_dict=None, weights_dict=None, mask_threshold_ratio_factor: float = 0.5):

        now = datetime.now()
        date = now.strftime("%Y_%m_%d_%H_%M_%S")
        preprcessing_path = date + new_ann_path

        self.dawid_skene = LocalizationAwareEM(preprcessing_path, annotator_key, iou_threshold, merging_ops, mask_threshold_ratio_factor, return_confusion_matrix=True)

        super(ExpectationMaximizationWBF, self).__init__(new_ann_path, annotator_key, iou_threshold, confidence_threshold,
                                                         scores_dict, weights_dict, mask_threshold_ratio_factor)

    def __call__(self, ann_file):
        if path.exists(self.new_ann_path):
            print("Using existing {} coco file at {}".format(self.__class__.__name__ ,self.new_ann_path))
            return self.new_ann_path

        self.dawid_skene(ann_file)

        # Convert confusion matrix into score_dict format
        annotations = self.dawid_skene.annotators
        classes = self.dawid_skene.classes
        error_rates = self.dawid_skene.error_rates

        self.scores_dict = self.convert_to_dict(annotations, classes, error_rates)

        return super(ExpectationMaximizationWBF, self).__call__(ann_file)


    def convert_to_dict(self, annotators, classes, error_rates):
        score_dict = {}
        for i, annotator in enumerate(annotators):
            score_dict[annotator] = {}
            for j, class_id in enumerate(classes):
                score_dict[annotator][class_id] = error_rates[i][j][j]

        return score_dict
