_base_ = "texbig_dataset.py"

data = dict(
    train=dict(  # Train dataset configs
        ann_pre_processing_fn=dict(
            type="PassthroughPre",
        ),
    )
)