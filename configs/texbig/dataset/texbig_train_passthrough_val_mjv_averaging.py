_base_ = "base/texbig_train_passthrough.py"

data = dict(
    val=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="texbig_val_mjv_averaging.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="averaging"
            )
        ),
    test=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="texbig_test_mjv_averaging.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="averaging"
        )
    )
)