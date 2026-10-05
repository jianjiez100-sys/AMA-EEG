# SEED fusion1 alignment

These frozen text and image projectors are the `fusion1` weights used for the
reported SEED experiments. They map raw 1024-dimensional CLIP features into a
shared feature space before AMA-EEG applies its trainable residual projectors.

To retrain the alignment weights, update the feature paths near the top of
`train_text_image_alignment.py` and run:

```bash
python multimodel_fusion/seed_fusion1/train_text_image_alignment.py
```

SEED pretraining requires the generated projected arrays. After extracting the
released raw feature archives under `features/SEED/`, generate these arrays with
the existing weights from the repository root:

```bash
python project_seed_features.py data=SEED
```

This projects each one-second vector before window averaging, as in the original
SEED experiment. The generated `projected_text/` and `projected_image/`
directories are ignored by Git. Retraining the alignment weights is optional
and is not needed for reproducing this preprocessing stage.

Projector checksums:

| File | SHA-256 |
| --- | --- |
| `projector_image.pt` | `42725a5cc30930ebc7776345b4554c5f6fd5d5c3711f8789eb9102f9cb7a5d83` |
| `projector_text.pt` | `195d25137b264aa5aa2a033a9f94c72fb8fe2c7c255c7dc1387d3b6d71faec47` |
