from collections import defaultdict
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.api_wrappers import COCO

from .basedataset import BaseDataset

@DATASETS.register_module()
class BaseDatasetCollect(BaseDataset):
    """
        This dataset is mostly the same as the CocoDataset (inheriting from CustomDataset and BaseDataset).
        The main difference is that image_id's are not what is loaded but file_names.

        The elements: (in method load_annotations)
            - data_infos
            - annotators_to_id
            - id_to_annotators
        are added.

        The methods load_annoations, get_ann_info and _parse_ann_info are overwriten to handle repeated labels in a
        proper way.

        To process the data in the pipeline as explained here: https://mmdetection.readthedocs.io/en/latest/tutorials/data_pipeline.html
        a function similar to LoadAnnotations (LoadRepeatedLabels) and DefaultFormatBundle (DefaultFormatBundleRepeatedLabels)
        are added. Without these functions the data is not formatted in a suitable way to be used for batch training.
        During the Collect stage of the pipeline "annotation" also needs to be added - otherwise it won't appear

        This Type of dataset cannot be used for evaluation since the possible ground truth is ambiguous.
        You should instead use the Datasets build on the BaseDataset, even for multi-annotated data and preprocess
        the annotations.
    """
    ANNOTATOR_KEY = None

    def load_annotations(self, ann_file):
        """
            Change loading of the data_infos to a multi-annotator variant, data infos should hold the following information:
            str: file_name - same as in the original annotation
            int: width - ""
            int: height - ""
            str: license: - ""
            list (int): ids - contains a list of ids for the different images - different as in original
            list (str): annotator - contains a list of annotator names - different as in original
        """
        assert self.annotator_key is not None, "Annotator key cannot be None, the class cannot be used without" \
                                                             "this parameters, please add it to code - this is not a hyperparameter."
        ann_file = self.ann_pre_processing_fn(ann_file)
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos_per_file_name = defaultdict(list)
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos_per_file_name[info['filename']].append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        data_infos = []
        annotators = set()
        for data_info_list in data_infos_per_file_name.values():
            data_info_new = data_info_list[0]
            assert "id" in data_info_new, "Key id is missing in {}, but should be there.".format(data_info_new)
            assert self.annotator_key in data_info_new, "Annotator key {} is missing in {}, but should be there.".format(self.annotator_key, data_info_new)
            data_info_new["coders"] = []
            data_info_new["ids"] = []
            for data_info in data_info_list:
                data_info_new["coders"].append(data_info[self.annotator_key])
                data_info_new["ids"].append(data_info["id"])
                annotators.add(data_info[self.annotator_key])
            del data_info_new[self.annotator_key]
            del data_info_new["id"]  # this should crash if the key doesn't exist and is intended.
            data_infos.append(data_info_new)
        self.annotators_to_id = {annotator: id for id, annotator in enumerate(sorted(annotators))}
        self.id_to_annotators = {id: annotator for id, annotator in enumerate(sorted(annotators))}
        return data_infos

    def get_ann_info(self, idx):
        """ Get COCO annotation by index.

            Different then the original function since each data_infos idx contains multiple image_ids.
            The annotations need to get the annotator included into their dictonary as well.
        """
        img_ids = self.data_infos[idx]["ids"]
        id_to_ann = {id: ann for id, ann in zip(self.data_infos[idx]["ids"], self.data_infos[idx]["coders"])}
        ann_ids = self.coco.get_ann_ids(img_ids=img_ids)
        ann_info = self.coco.load_anns(ann_ids)
        for ann in ann_info:
            ann["coder"] = self.annotators_to_id[id_to_ann[ann["image_id"]]]
        data_info, ann_info  = self.ann_prep_fn(self.data_infos[idx], ann_info)
        return self._parse_ann_info(data_info, ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """ Parse bbox and mask annotation.

            Different then the original functions since it also includes information regarding the annotator.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_coder_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_coder_ann.append(ann["coder"])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_coder_ann = np.array(gt_coder_ann, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_coder_ann = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            coder=gt_coder_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        raise Exception("It is not possible to evaluate a metric on ambiguous ground truth, please use a different"
                        "Dataset that can provide this functionality.")


