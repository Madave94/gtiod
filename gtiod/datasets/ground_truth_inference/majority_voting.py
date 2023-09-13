import sys
from collections import defaultdict, Counter
from itertools import combinations, product
from copy import deepcopy
from random import choice

from .builder import GTINFERENCE
from .base_gt_aggregator import BaseAggregator

@GTINFERENCE.register_module()
class MajorityVoting(BaseAggregator):
    def __init__(self, new_ann_path: str, annotator_key: str, iou_threshold: float, merging_ops: str, mask_threshold_ratio_factor: float=0.5 ):
        """ Majority Voting method that uses the bounding box with the highest agreement.
            The agreement needs to be at least half the number of annotators that labeled the current image, for the edge
            case of exactly 50%, the chance of the annotation being included is also exactly 50%.

        :param new_ann_path: the location where the new annotation file is stored it. It will be stored OS independetly in
               the temporary folder, so after reboot the file would be gone.
        :param annotator_key: For VinDR-CXR this should be "rad_id" and for texbig "annotator".
        :param iou_threshold: the minimum threshold at which annotations are considered overlapping.
        :param merging_ops: union or intersection. Are boxes combined and this is the result of the MJV or is only the
                intersecting area taken.
        :param mask_threshold_ratio_factor: Threshold that is multiplied with the iou_threshold. This is the minimum mask
                overlap for  annotations to be considered AFTER the iou_threshold already matched. The parameter should
                be between 1 - 0.1. Lower values might lead to problems since there is no valid polygon that can be
                created. High values might lead to no matches, since it is harder to achieve high IoU's then bbox overlaps.
        """
        assert sys.version_info >= (3, 7), "Python version needs to be at least 3.7 or higher. Since majority voting relies " \
                                            "on ordered dictonaries that have been introduced at 3.7, majority voting will not " \
                                            "provide reliable results."
        new_ann_path = "iou{}_{}_".format(iou_threshold, merging_ops) + new_ann_path
        super(MajorityVoting, self).__init__(new_ann_path, annotator_key, iou_threshold)
        self.mask_threshold_ratio_factor = iou_threshold * mask_threshold_ratio_factor # this sets a threshold for the segmentation, needs to be at least 0.10
        assert self.mask_threshold_ratio_factor >= 0.10, "Mask sizes should have at least a IoU of 0.1 to be valid. Otherwise this might not produce valid polygons."
        assert merging_ops.lower() in ["union", "intersection", "averaging"], "Merging operator parameter wrong. {} is not allowed".format(merging_ops)
        self.merging_ops = merging_ops.lower() # valid merging ops should be intersection, union, averaging, median

    def process_image_annotations(self, images, annotations):
        """ Outer loop for a single image to execute the majority voting.
            Takes the images and annotations of the annotators and returns an image with a new id and the majority
            voted annotations (also with a new id).
        """
        number_of_annotators = len(images)
        min_agreement = number_of_annotators/2.0 # minimum number of annotators to agree on an element to be a mj voted gt
        new_annotations = []

        # create new image with a new unique id
        new_image = deepcopy(next(iter(images.values())))
        new_image_id = next(self.new_img_id_generator)
        new_image["id"] = new_image_id
        new_image[self.annotator_key] = "Majority Voting"
        width = new_image["width"]
        height = new_image["height"]

        # try to find matching annotations starting with the most annotator going to the least possible number
        # of annotators
        while number_of_annotators >= min_agreement and len(annotations) > 0:
            coder_to_annotations_dict = defaultdict(list)
            for annotation in annotations:
                coder_to_annotations_dict[annotation[self.annotator_key]].append(annotation)
            # create combinations
            current_combinations = list(combinations(coder_to_annotations_dict, number_of_annotators))
            # select combination annotations - this might return a huge list
            boxes_to_fuse = []
            for combination in current_combinations:
                if len(combination) > 1:
                    current_annotation_combinations = list(product(*[coder_to_annotations_dict[annotator] for annotator in combination]))
                else:
                    current_annotation_combinations = coder_to_annotations_dict[combination[0]]
                boxes_to_fuse += current_annotation_combinations
            # retrieve newest mjv annotations and remove the ones used
            # --- this is the core logic of the majority voting ---
            fused_boxes = self.voting(boxes_to_fuse, min_agreement, height, width)
            # add annotations to the mjv annotations and exclude such selected annotations from the future mjv process
            # this uses the property that dictionaries are ordered since python 3.7
            remaining_ids = set([annotation["id"] for lst in coder_to_annotations_dict.values() for annotation in lst])
            for iou_and_ids, mjv_ann in reversed(fused_boxes.items()):
                ids = set(iou_and_ids[1:])
                if ids.issubset(remaining_ids):
                    mjv_ann["image_id"] = new_image_id
                    new_annotations.append(mjv_ann)
                    remaining_ids = remaining_ids.difference(ids)
            annotations = list(filter(lambda annotation: annotation["id"] in remaining_ids, annotations))
            # decrease the number of annotator by one
            number_of_annotators -= 1

        return new_image, new_annotations

    def voting(self, boxes_to_fuse, min_agreement, height, width):
        """ Here the voting procedure is executed
        """
        matches = {}
        for boxes in boxes_to_fuse:
            if isinstance(boxes, tuple):
                iou_bbox = self.iou_bbox(boxes)
                if iou_bbox < self.iou_threshold:
                    continue
                if self.iou_segm(boxes, height, width) < self.mask_threshold_ratio_factor:
                    continue
                class_counts = Counter([annotation["category_id"] for annotation in boxes])
                category_id, frequency = class_counts.most_common(1)[0]
                # edge case for exactly 50%
                if frequency == min_agreement:
                    # check if there is a second class with the same frequency
                    choices = [category_id]
                    if len(class_counts) > 1:
                        category_id_2, frequency_2 = class_counts.most_common(2)[1]
                        if frequency_2 == min_agreement:
                            choices.append(category_id_2)
                    category_id = choice(choices)
                # after adding the edge case proceed normally
                if frequency >= min_agreement:
                    ann_ids = [box["id"] for box in boxes]
                    matches[tuple([iou_bbox] + ann_ids)] = self.merge_annotations(boxes, category_id, height, width)
            else:
                # single annoation, edge cases
                if min_agreement < 1.0:
                    add_annotation = True
                elif min_agreement == 1.0:
                    add_annotation = choice([True, False])
                if add_annotation:
                    new_annotation = deepcopy(boxes)
                    new_ann_id = next(self.new_annotation_id_generator)
                    new_annotation["id"] = new_ann_id
                    new_annotation[self.annotator_key] = "Majority Voting"
                    # set minimum threshold in case it matched
                    matches[tuple([self.iou_threshold] + [boxes["id"]])] = new_annotation
        return matches

    def merge_annotations(self, annotations, category_id, height, width):
        votingAnnotators = {}
        annLabel = []
        annLabel2 = []

        annLabel.append(annotations[0]["category_id"])
        annLabel2.append(annotations[1]["category_id"])

        votingAnnotators[annotations[0][self.annotator_key]] = annLabel
        votingAnnotators[annotations[1][self.annotator_key]] = annLabel2

        annA = annotations[0]

        if self.merging_ops == "union":
            new_bbox = annA["bbox"]
            new_segm = self.get_segmentation_mask(annA, height, width)
        if self.merging_ops == "intersection":
            new_bbox = annA["bbox"]
            new_segm = self.get_segmentation_mask(annA, height, width)
        for annB in annotations[1:]:
            if self.merging_ops == "union":
                new_bbox = self.union_bbox(new_bbox, annB["bbox"])
                new_segm = new_segm.union(self.get_segmentation_mask(annB, height, width))
            if self.merging_ops == "intersection":
                new_bbox, _ = self.intersection_bbox_and_area(new_bbox, annB["bbox"])
                new_segm = new_segm.intersection(self.get_segmentation_mask(annB, height, width))

        if self.merging_ops == "averaging":
            bbox_conf_list = [[ann["bbox"], 1.0] for ann in annotations]
            _, new_bbox = self.get_weighted_box(bbox_conf_list, width, height)
            segm_conf_list = [[self.get_segmentation_mask(ann, height, width), 1.0] for ann in annotations]
            new_segm, _ = self.get_weighted_segmentation(segm_conf_list)

        new_annotation = deepcopy(annA)
        new_ann_id = next(self.new_annotation_id_generator)
        new_annotation["id"] = new_ann_id
        new_annotation["bbox"] = new_bbox
        if self.merging_ops == "averaging":
            new_annotation["segmentation"] = new_segm
        else:
            new_annotation["segmentation"] = self.get_coco_segmentation(new_segm)
        new_annotation["category_id"] = category_id
        new_annotation[self.annotator_key] = "Majority Voting"

        return new_annotation


