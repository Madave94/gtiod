from mmdet.datasets.builder import DATASETS

from gtiod.datasets.basedataset import BaseDataset

@DATASETS.register_module()
class TexBiGDataset(BaseDataset):
    """
        This is mainly a copy of the COCO Dataloader mmdet.datasets.CocoDataset
    """
    CLASSES = ("paragraph", "equation", "logo", "editorial note", "sub-heading", "caption", "image",
           "footnote", "page number", "table", "heading", "author", "decoration", "footer",
           "header", "noise", "frame", "column title", "advertisement")

    PALETTE = [(61, 61, 245), (128,128,0), (51, 221, 255), (250, 50, 83), (170, 240, 209), (131, 224, 112),
               (36, 179, 83), (255, 204, 51), (89, 134, 179), (128,0,128), (42, 125, 209), (255, 0, 204),
               (255, 96, 55),(50, 183, 250), (66, 201, 18), (255, 0, 0), (184, 61, 245), (102, 0, 255),(102, 0, 51)]