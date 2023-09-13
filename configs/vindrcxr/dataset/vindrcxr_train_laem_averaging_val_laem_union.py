_base_ = "base/vindrcxr_train_laem_averaging.py"

data = dict(
    val=dict(
        ann_pre_processing_fn=dict(
            type="LocalizationAwareEM",
            new_ann_path="vindrcxr_val_laem_union.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            merging_ops="union"
        )
    ),
    test=dict(
        ann_pre_processing_fn=dict(
            type="LocalizationAwareEM",
            new_ann_path="vindrcxr_test_laem_union.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            merging_ops="union"
        )
    )
)