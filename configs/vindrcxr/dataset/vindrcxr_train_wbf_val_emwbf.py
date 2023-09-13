_base_ = "base/vindrcxr_train_wbf.py"

data=dict(
    val=dict(
        ann_pre_processing_fn=dict(
            type="ExpectationMaximizationWBF",
            new_ann_path="vindrcxr_val_emwbf.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            merging_ops = "averaging"
        )
    ),
    test=dict(
        ann_pre_processing_fn=dict(
            type="ExpectationMaximizationWBF",
            new_ann_path="vindrcxr_test_emwbf.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            merging_ops = "averaging"
        )
    )
)