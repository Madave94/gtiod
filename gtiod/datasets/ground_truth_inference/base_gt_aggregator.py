import json
import tempfile
from os import path
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np

from gtiod.datasets.utils import SegmentationMask

class BaseAggregator():
    """ Implement the basic implementation logic shared among all methods that somehow aggregate noisy repeated labels
        to an approximated ground truth label.

        Data should be in coco format.

        The class and call should include all kinds of functionality that are shared by all sub-classes to prevent
        unnecessary boilerplate code. The structure should in no way restrict the flexibility how classes are implemented.
        Furthermore, this class is not supposed to be loaded within a config file since it does not provide usable
        code.
    """

    def __init__(self, new_ann_path: str, annotator_key: str, iou_threshold: float):
        self.new_ann_path = path.join(tempfile.gettempdir(), new_ann_path)
        self.annotator_key = annotator_key
        self.iou_threshold = iou_threshold
        self.new_img_id_generator = self.id_generator()
        self.new_annotation_id_generator = self.id_generator()

    def __call__(self, ann_file):
        # link to the existing annotation file, if it was already created.
        if path.exists(self.new_ann_path):
            print("Using existing {} coco file at {}".format(self.__class__.__name__ ,self.new_ann_path))
            return self.new_ann_path

        # create new annotation file.
        with open(ann_file, "r") as f:
            coco_annotations = json.load(f)
        print("Creating new {} coco file from {} and saved to {}.".format(self.__class__.__name__, ann_file, self.new_ann_path))

        coco_annotations = self.pre_process_annotations(coco_annotations)

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
        new_images = []
        new_annotations = []
        for file_name in tqdm(file_name_to_annotations_dict.keys()):
            image, annotations = self.process_image_annotations(
                file_name_to_image_dict[file_name],
                file_name_to_annotations_dict[file_name]
            )
            new_images.append(image)
            new_annotations += annotations
        coco_annotations["annotations"] = new_annotations
        coco_annotations["images"] = new_images

        coco_annotations = self.post_process_annotations(coco_annotations)

        with open(self.new_ann_path, "w") as f:
            json.dump(coco_annotations, f)

        return self.new_ann_path

    def pre_process_annotations(self, coco_annotations):
        # Function hook for pre-processing of annotations
        return coco_annotations

    def process_image_annotation(self, images, annotations):
        raise NotImplementedError()

    def post_process_annotations(self, coco_annotations):
        # Function hook for post-processing of annotations
        return coco_annotations

    def iou_bbox(self, boxes):
        intersect_box = boxes[0]["bbox"]
        union_box = boxes[0]["bbox"]
        for annotation in boxes[1:]:
            intersect_box, intersection_area = self.intersection_bbox_and_area(intersect_box, annotation["bbox"])
            if intersection_area == 0.0:
                return 0.0
            union_box = self.union_bbox(union_box, annotation["bbox"])
        iou = intersection_area / self.get_bbox_area(union_box)
        assert iou <= 1
        return iou

    def iou_segm(self, boxes, height, width):
        maskA = self.get_segmentation_mask(boxes[0], height, width)
        intersect_box = maskA
        union_box = maskA
        for annotation in boxes[1:]:
            maskB = self.get_segmentation_mask(annotation, height, width)
            intersect_box = intersect_box.intersection(maskB)
            if intersect_box.get_area() == 0.0:
                return 0.0
            union_box = union_box.union(maskB)
        iou = intersect_box.get_area() / union_box.get_area()
        assert iou <= 1
        return iou

    def intersection_bbox_and_area(self, box_a, box_b):
        """
        :param box_a: xywh bbox
        :param box_b: xywh bbox
        :return: (area, xywh bbox)
        """
        box_a = [box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]]
        box_b = [box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]]
        x_A = max(box_a[0], box_b[0])
        y_A = max(box_a[1], box_b[1])
        x_B = min(box_a[2], box_b[2])
        y_B = min(box_a[3], box_b[3])
        area = abs(max(x_B - x_A, 0) * max(y_B - y_A, 0))
        if area == 0:
            return None, area
        else:
            new_box = [x_A, y_A, x_B - x_A, y_B - y_A]
            return new_box, area

    def union_bbox(self, box_a, box_b):
        """
        :param box_a: xywh bbox
        :param box_b: xywh bbox
        :return: xywh bbox
        """
        box_a = [box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]]
        box_b = [box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]]
        x_A = min(box_a[0], box_b[0])
        y_A = min(box_a[1], box_b[1])
        x_B = max(box_a[2], box_b[2])
        y_B = max(box_a[3], box_b[3])
        new_box = [x_A, y_A, x_B - x_A, y_B - y_A]
        return new_box

    def get_bbox_area(self, box):
        """
        :param box: xywh bbox
        :return: area of that box
        """
        area = box[2] * box[3]
        return area

    def get_segmentation_mask(self, segmentation, height, width):
        return SegmentationMask(segmentation, height, width)

    def get_coco_segmentation(self, segmentation_mask):
        return segmentation_mask.maskToSeg()

    def get_weighted_box(self, box_list, img_width, img_height):
        """
            box_list: [
                        [[x,y,w,h], weight],
                        [[x,y,w,h], weight],
                        ...                ,
                      ]
            img_width & img_height is needed to secure that compute bounding boxes are possible due to numeric instabilities.
        """
        bbox = np.zeros(4)
        scores = 0

        for box, score in box_list:
            scores += score
            box = np.array(box)
            bbox[:] += (score * box[:])

        conf = scores / len(box_list)
        bbox = bbox[:] / scores
        bbox = [float(coord) for coord in bbox]

        # floating point errors might cause bboxes bigger then the actual image, this will round them down.
        # this corrects the error, if it is just of a numeric nature, if there is in fact an error it will throw the assert.
        if box[0]+box[2] > img_width:
            box[0] = math.floor(box[0])
            box[2] = math.floor(box[2])
        if box[1]+box[3] > img_height:
            box[1] = math.floor(box[1])
            box[3] = math.floor(box[3])
        assert box[0]+box[2] <= img_width, "Invalid bounding box {}, width is larger then image width {}.".format(box, img_width)
        assert box[1]+box[3] <= img_height, "Invalid bounding box {}, height is larger then image height {}.".format(box, img_height)

        return conf, bbox

    def get_weighted_segmentation(self, mask_list):
        """
            mask_list: [
                          (mask, weight),
                          (mask, weight),
                          ...           ,
                       ]
        """
        weighted_area = 0
        weighted_center = np.zeros(2)
        scores = 0
        for mask_entry in mask_list:
            mask, score = mask_entry
            scores += score
            mask_area = mask.get_area()
            weighted_area += mask_area * score
            center_point = mask.get_maskCenter()
            weighted_center[:] += (np.array(center_point)[:] * score)
            mask_entry.append(center_point)

        target_area = weighted_area / scores
        target_cp = list(weighted_center[:] / scores)

        # Select mask with the closest distance to the weighted "target center point" for dilation/erosion
        chosen_mask = None
        min_dist = math.inf
        for mask, score, center_point in mask_list:
            dist = math.dist(center_point, target_cp)
            if dist < min_dist:
                min_dist = dist
                chosen_mask = mask

        assert chosen_mask is not None, "Cannot continue processing without a mask."

        # Dilation/erosion of the chosen mask until the mask reaches the weighted "target area"
        kernel = np.ones((3, 3))
        chosen_mask_area = chosen_mask.get_area()
        if chosen_mask_area > target_area:
            while chosen_mask.get_area() > target_area:
                chosen_mask = chosen_mask.erosion(kernel)

            return chosen_mask.maskToSeg(), chosen_mask.get_area()

        elif chosen_mask_area < target_area:
            while chosen_mask.get_area() < target_area:
                chosen_mask = chosen_mask.dilation(kernel)

            return chosen_mask.maskToSeg(), chosen_mask.get_area()

        else:
            return chosen_mask.maskToSeg(), chosen_mask.get_area()

    def id_generator(self):
        id = 0
        while True:
            yield id
            id += 1