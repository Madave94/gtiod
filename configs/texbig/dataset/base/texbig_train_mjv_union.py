_base_ = "texbig_dataset.py"

data = dict(
    train=dict(  # Train dataset config
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="texbig_train_mjv_union.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="union"
            )
        )
    )