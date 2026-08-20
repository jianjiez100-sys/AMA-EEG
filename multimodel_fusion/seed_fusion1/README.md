# SEED fusion1 alignment

These frozen text and image projectors are the `fusion1` weights used for the
reported SEED experiments. They map raw 1024-dimensional CLIP features into a
shared feature space before AMA-EEG applies its trainable residual projectors.

To retrain the alignment weights, update the feature paths near the top of
`train_text_image_alignment.py` and run:

```bash
python multimodel_fusion/seed_fusion1/train_text_image_alignment.py
```

The generated projected feature arrays are intermediate artifacts and are not
required by AMA-EEG. Keep using raw features with the saved projector weights.
