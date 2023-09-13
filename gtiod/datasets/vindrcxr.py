from mmdet.datasets.builder import DATASETS

from gtiod.datasets.basedataset import BaseDataset

@DATASETS.register_module()
class VinDRCXRDataset(BaseDataset):
    CLASSES = ("Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD",
               "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
               "Pneumothorax", "Pulmonary fibrosis")

    PALETTE = [(61, 61, 245), (128, 128, 0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
               (36, 179, 83), (255, 204, 51), (89, 134, 179), (128, 0, 128), (42, 125, 209), (255, 0, 204),
               (255, 96, 55), (50, 183, 250)]