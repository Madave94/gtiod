_base_ = "base/texbig_train_laem_intersection.py"

data = dict(
    val=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="LocalizationAwareEM",
            new_ann_path="texbig_val_laem_union.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="union"
            )
        ),
    test=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="LocalizationAwareEM",
            new_ann_path="texbig_test_laem_union.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="union"
        )
    )
)