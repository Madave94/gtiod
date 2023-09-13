_base_ = "base/vindrcxr_train_mjv_union.py"

data = dict(
    val=dict(
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="vindrcxr_val_mjv_union.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            merging_ops="union"
        )
    ),
    test=dict(
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="vindrcxr_test_mjv_union.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            merging_ops="union"
        )
    )
)