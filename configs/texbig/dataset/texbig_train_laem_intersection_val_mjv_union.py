_base_ = "base/texbig_train_laem_intersection.py"

data = dict(
    val=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="texbig_val_mjv_union.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="union"
            )
        ),
    test=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="texbig_test_mjv_union.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="union"
        )
    )
)