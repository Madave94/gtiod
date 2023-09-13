_base_ = "base/vindrcxr_train_laem_union.py"

data = dict(
    val=dict(
        ann_pre_processing_fn=dict(
            type="WeightedBoxesFusion",
            new_ann_path="vindrcxr_val_wbf.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            scores_dict=None,
            weights_dict=None
        )
    ),
    test=dict(
        ann_pre_processing_fn=dict(
            type="WeightedBoxesFusion",
            new_ann_path="vindrcxr_val_wbf.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            scores_dict=None,
            weights_dict=None
        )
    )
)