_base_ = "vindrcxr_dataset.py"

data = dict(
    train=dict(
        ann_pre_processing_fn=dict(
            type="MajorityVoting",
            new_ann_path="vindrcxr_train_mjv_union.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            merging_ops="union"
        )
    )
)
    