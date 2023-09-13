from copy import deepcopy
from collections import defaultdict

from .builder import GTINFERENCE
from .base_gt_aggregator import BaseAggregator

@GTINFERENCE.register_module()
class WeightedBoxesFusion(BaseAggregator):

    def __init__(self, new_ann_path, annotator_key, iou_threshold, confidence_threshold,
                 scores_dict=None, weights_dict=None, mask_threshold_ratio_factor: float =0.5):
        """Weighted Boxes Fusion method uses confidence scores of all annotations to calculate the average annotation.
           The method won't discard any annotations, instead, it combines them.

           :param new_ann_path: the location where the new annotation file is stored it. It will be stored OS
                  independetly in the temporary folder, so after reboot the file would be gone.
           :param annotator_key: For VinDR-CXR this should be "rad_id" and for texbig "annotator".
           :param iou_threshold: the minimum threshold at which annotations are considered overlapping.
           :param confidence_threshold: the minimum threshold at which annotations are considered in the algorithm.
           :param scores_dict: dictionary containing the confidence scores for each annotator and label. If the
                  parameter is None, the confidence score is set to 1.0 for all annotations.
           :param weights_dict: dictionary containing the weights for each annotator. If the parameter is None, the
                  weight is set to 1.0 for all annotators.
           """
        super(WeightedBoxesFusion, self).__init__(new_ann_path, annotator_key, iou_threshold)
        self.confidence_threshold = confidence_threshold
        self.scores_dict = scores_dict
        self.weights_dict = weights_dict
        self.mask_threshold_ratio_factor = iou_threshold * mask_threshold_ratio_factor

    def pre_process_annotations(self, coco_annotations):
        if self.scores_dict is not None:
            new_images = []
            new_annotations = []
            included_img_ids = set()
            for image in coco_annotations["images"]:
                if image[self.annotator_key] in self.scores_dict:
                    included_img_ids.add(image["id"])
                    new_images.append(image)

            for annotation in coco_annotations["annotations"]:
                if annotation["image_id"] in included_img_ids:
                    new_annotations.append(annotation)

            print(f"A total of {len(coco_annotations['images'])} images are contained in the original annotation file")
            print(f"A total of {len(coco_annotations['annotations'])} annotations are contained in the original annotation file")

            coco_annotations["annotations"] = new_annotations
            coco_annotations["images"] = new_images

            print(f"A total of {len(coco_annotations['images'])} images are contained in the new annotation file")
            print(f"A total of {len(coco_annotations['annotations'])} annotations are contained in the new annotation file")

        if self.scores_dict is None or self.weights_dict is None:
            file_name_to_annotations_dict = defaultdict(list)
            file_name_to_image_dict = defaultdict(dict)
            id_to_file_name = dict()

            for image in coco_annotations["images"]:
                file_name = image["file_name"]
                image_id = image["id"]
                file_name_to_image_dict[file_name][image_id] = image
                id_to_file_name[image_id] = file_name

            for annotation in coco_annotations["annotations"]:
                image_id = annotation["image_id"]
                file_name = id_to_file_name[image_id]
                annotator_name = file_name_to_image_dict[file_name][image_id][self.annotator_key]
                annotation[self.annotator_key] = annotator_name
                file_name_to_annotations_dict[file_name].append(annotation)

            # create a dictionary for the annotator weights, if the dictionary does not exist.
            if self.weights_dict is None:
                self.weights_dict = {}
                for img_name, annotations in file_name_to_annotations_dict.items():
                    for annotation in annotations:
                        if annotation[self.annotator_key] not in self.weights_dict:
                            self.weights_dict[annotation[self.annotator_key]] = 1.0

            # create a dictionary for the confidence scores of the annotations, if the dictionary does not exist.
            if self.scores_dict is None:
                self.scores_dict = {}
                for img_name, annotations in file_name_to_annotations_dict.items():
                    for annotation in annotations:
                        annotator = annotation[self.annotator_key]
                        if annotator not in self.scores_dict:
                            self.scores_dict[annotator] = {}
                            self.scores_dict[annotator][annotation["category_id"]] = 1.0
                        elif annotation["category_id"] not in self.scores_dict[annotator]:
                            self.scores_dict[annotator][annotation["category_id"]] = 1.0

        return coco_annotations

    def process_image_annotations(self, images, annotations):
        """Outer loop for a single image to execute the weighted boxes fusion.
            Takes the images and annotations of the annotators and returns an image with a new id and the fused
            annotations (also with a new id).
        """

        # sort the annotations according to the annotators
        coder_to_annotations_dict = defaultdict(list)
        for annotation in annotations:
            coder_to_annotations_dict[annotation[self.annotator_key]].append(annotation)

        # create a new image with a new unique id
        new_image = deepcopy(next(iter(images.values())))
        new_img_id = next(self.new_img_id_generator)
        new_image["id"] = new_img_id
        new_image[self.annotator_key] = "wbf"
        img_width = new_image["width"]
        img_height = new_image["height"]

        gt_annotations = self.weighted_boxes_fusion(coder_to_annotations_dict, img_width, img_height)

        # format the calculated ground truth annotations
        for gt_annotation in gt_annotations:
            new_ann_id = next(self.new_annotation_id_generator)
            gt_annotation["id"] = new_ann_id
            gt_annotation["image_id"] = new_img_id
            gt_annotation.pop(self.annotator_key)

        return new_image, gt_annotations

    def weighted_boxes_fusion(self, annotation_dict, img_width, img_height):
        """Calculation of the weighted boxes fusion algorithm for multi-label annotations.
           :param annotation_dict: Contains all image annotations with the annotator as key and the corresponding
                  annotations in a list.
           :param img_width: Image width.
           :param img_height: Image height.
           :return annotations: fused annotations
        """

        # add the confidence score per annotation and sum up all
        # weights for rescaling the confidence scores later.
        sum_of_weights = 0
        for annotator, annotations in annotation_dict.items():
            weight = self.weights_dict[annotator]
            sum_of_weights += weight
            for annotation in annotations:
                score = weight * self.scores_dict[annotator][annotation["category_id"]]
                annotation["conf_score"] = score

        # sort the annotations by label.
        label_to_annotation_dict = defaultdict(list)
        for annotator, annotations in annotation_dict.items():
            for annotation in annotations:
                label_to_annotation_dict[annotation["category_id"]].append(annotation)

        overall_annotations = []
        for label, annotations in label_to_annotation_dict.items():

            # Sort annotations per label by confidence score
            sorted_annotations = sorted(annotations, key=lambda d: d["conf_score"], reverse=True)

            clustered_annotations = []
            fused_annotations = []

            for annotation in sorted_annotations:
                index = self.find_matching_annotation(img_width, img_height, fused_annotations, annotation)

                if index != -1:
                    clustered_annotations[index].append(annotation)
                    fused_annotations[index] = self.get_fused_annotation(clustered_annotations[index], img_width, img_height)

                else:
                    clustered_annotations.append([annotation])
                    fused_annotations.append(deepcopy(annotation))

            # rescale confidence scores based on number of models and annotations.
            for idx in range(len(clustered_annotations)):
                fused_annotations[idx]["conf_score"] = fused_annotations[idx]["conf_score"] * \
                                                       len(clustered_annotations[idx]) / sum_of_weights

            overall_annotations += fused_annotations

        return overall_annotations

    def find_matching_annotation(self, img_width, img_height, annotation_list, new_annotation):
        max_iou = 0
        index = -1
        for idx, annotation in enumerate(annotation_list):
            box_iou = self.iou_bbox([annotation, new_annotation])
            seg_iou = self.iou_segm([annotation, new_annotation], img_height, img_width)
            iou = (box_iou + seg_iou) / 2.0

            if (box_iou >= self.iou_threshold) and (seg_iou >= self.mask_threshold_ratio_factor) and (iou > max_iou):
                index = idx
                max_iou = iou

        return index

    def get_fused_annotation(self, annotations, img_width, img_height):
        box_list = []
        seg_list = []
        for annotation in annotations:
            box_list.append([annotation["bbox"], annotation["conf_score"]])
            mask = self.get_segmentation_mask(annotation, img_height, img_width)
            seg_list.append([mask, annotation["conf_score"]])

        score, box = self.get_weighted_box(box_list, img_width, img_height)
        segmentation, area = self.get_weighted_segmentation(seg_list)

        new_annotation = deepcopy(annotations[0])

        new_annotation["bbox"] = box
        new_annotation["conf_score"] = score
        new_annotation["segmentation"] = segmentation
        new_annotation["area"] = int(area)

        return new_annotation