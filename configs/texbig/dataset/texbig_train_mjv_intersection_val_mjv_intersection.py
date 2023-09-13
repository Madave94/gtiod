_base_ = "base/texbig_train_mjv_intersection.py"

data = dict(
    val=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="texbig_val_mjv_intersection.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="intersection"
            )
        ),
    test=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="texbig_test_mjv_intersection.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="intersection"
        )
    )
)