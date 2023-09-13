_base_ = "vindrcxr_dataset.py"

data = dict(
    train=dict(
        ann_pre_processing_fn=dict(
            type="ExpectationMaximizationWBF",
            new_ann_path="vindrcxr_train_emwbf.json",
            annotator_key="rad_id",
            iou_threshold=0.5,
            confidence_threshold=0.0001,
            merging_ops = "averaging"
        )
    )
)