_base_ = "texbig_dataset.py"

data = dict(
    train=dict(  # Train dataset configs
        ann_pre_processing_fn=dict(
            type="LocalizationAwareEM",
            new_ann_path="texbig_train_laem_union.json",
            annotator_key="annotator",
            iou_threshold=0.5,
            merging_ops="union",
        ),
    )
)