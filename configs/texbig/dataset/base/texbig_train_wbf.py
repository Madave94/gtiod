_base_ = "texbig_dataset.py"

data = dict(
    train=dict(  # Train dataset configs
        ann_pre_processing_fn=dict(
            type="WeightedBoxesFusion",
            new_ann_path="texbig_train_wbf.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            scores_dict=None,
            weights_dict=None
        ),
    )
)