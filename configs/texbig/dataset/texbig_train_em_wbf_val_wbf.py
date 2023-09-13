_base_ = "base/texbig_train_em_wbf.py"

data = dict(
    val=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="WeightedBoxesFusion",
            new_ann_path="texbig_val_wbf.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            scores_dict=None,
            weights_dict=None
            )
        ),
    test=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="WeightedBoxesFusion",
            new_ann_path="texbig_test_wbf.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            scores_dict=None,
            weights_dict=None
        )
    )
)