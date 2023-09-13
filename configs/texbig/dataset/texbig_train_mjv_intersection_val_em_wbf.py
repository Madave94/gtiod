_base_ = "base/texbig_train_mjv_intersection.py"

data = dict(
    val=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="ExpectationMaximizationWBF",
            new_ann_path="texbig_val_em_wbf.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            merging_ops="intersection"
            )
        ),
    test=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="ExpectationMaximizationWBF",
            new_ann_path="texbig_test_em_wbf.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            merging_ops="intersection"
        )
    )
)