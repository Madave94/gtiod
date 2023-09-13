"""
    Contains Mask object, that should be used for union and intersection operation
"""
from pycocotools import mask as maskUtils
import numpy as np
import cv2

class SegmentationMask(object):
    def __init__(self, data, height=None, width=None):
        """
            Constructor data reads data as
            - RLE (Run-Length-Encoding)
            - Mask (binary segementation mask as a np.array)
            - Coco Annotations as str polygon

            The object should not be changed after initialization and operations like
            union or intersection should return a new SegmentationMask object.
        """
        # Find Type
        self.rle = None
        self.mask = None
        self.segmentation = None
        self.height = height
        self.width = width
        if self.isRLE(data):
            self.rle = data
        if self.isMask(data):
            self.mask = data
        if self.isAnnotation(data):
            self.segmentation = data["segmentation"]
            assert self.height is not None, "Need to provide height, when using annotation format."
            assert self.width is not None, "Need to provide width, when using annotation format."

    def isRLE(self, segm):
        if isinstance(segm, dict):
            if "counts" in segm:
                return True
        return False

    def isMask(self, segm):
        if isinstance(segm, np.ndarray):
            return True
        return False

    def isAnnotation(self, segm):
        if isinstance(segm, dict):
            if "segmentation" in segm:
                return True
        return False

    def get_mask(self):
        if self.mask is not None:
            return self.mask
        if self.rle is not None:
            self.mask = self.rleToMask(self.rle)
            return self.mask
        self.rle = self.segToRLE(self.segmentation, self.height, self.width)
        self.mask = self.rleToMask(self.rle)
        return self.mask

    def segToRLE(self, segm, h, w):
        """
            Args:
                h: int
                w: int
                segm: list(list, list) of float
            Return:
                rleObjs = {dict: 2}
                    "size" = {list: 2} [h, w]
                    "counts" = {str} 'rle ....'
        """
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
        return rle

    def rleToMask(self, rle):
        """
            Args rleObjs = {dict: 2}
                "size" = {list: 2} [h, w]
                "counts" = {str} 'rle ....'
            Returns:
                np.array with size (h, w) representing the presence of a class with 1 and the absence with 0
        """
        return maskUtils.decode(rle)

    def maskToSeg(self):
        """
            Args:
                np.array with size (h, w) representing the presence of a class with 1 and the absence with 0
            Returns:
                segm: list(list, list) of float
        """
        # Code from https://github.com/cocodataset/cocoapi/issues/476#issuecomment-871804850
        def polygonFromMask(maskedArr):
            # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
            contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            valid_poly = 0
            for contour in contours:
                # Valid polygons have >= 6 coordinates (3 points)
                if contour.size >= 6:
                    segmentation.append(contour.astype(float).flatten().tolist())
                    valid_poly += 1
            if valid_poly == 0:
                raise ValueError
            return segmentation
        return polygonFromMask(self.mask)

    def union(self, other_mask):
        own_mask = self.get_mask()
        other_mask = other_mask.get_mask()
        union = own_mask | other_mask
        return SegmentationMask(union)

    def intersection(self, other_mask):
        own_mask = self.get_mask()
        other_mask = other_mask.get_mask()
        intersection = own_mask & other_mask
        return SegmentationMask(intersection)

    def get_area(self):
        return self.get_mask().sum()

    def erosion(self, kernel, pad=1):
        area_before = self.get_area()
        mask = SegmentationMask(cv2.erode(self.get_mask(), kernel))
        if area_before == mask.get_area():
            mask = cv2.erode(np.pad(mask.get_mask(), pad), kernel)
            reduced_mask = self.unpad(mask, pad)
            mask = SegmentationMask(reduced_mask)
        return mask

    def unpad(self, mask, pad):
        dim_1, dim_2 = mask.shape
        new_mask = mask[pad:dim_1-pad, pad:dim_2-pad]
        return new_mask

    def dilation(self, kernel):
        mask = self.get_mask()
        return SegmentationMask(cv2.dilate(mask, kernel))

    def get_maskCenter(self):
        # Code from: https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
        mask = self.get_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0, "There is no contoure, which is necessary to find the center of a mask."
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
        else:
            return 0, 0
    