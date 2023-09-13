_base_ = "base/vindrcxr_train_laem_intersection.py"

data = dict(
    val=dict(
        ann_pre_processing_fn=dict(
            type="LocalizationAwareEM",
            new_ann_path="vindrcxr_val_laem_intersection.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            merging_ops="intersection"
        )
    ),
    test=dict(
        ann_pre_processing_fn=dict(
            type="LocalizationAwareEM",
            new_ann_path="vindrcxr_test_laem_intersection.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            merging_ops="intersection"
        )
    )
)