# FACED fusion10 alignment assets

This directory contains the frozen text/image alignment projectors used by the
FACED configuration and small stimulus-derived arrays/figures used to inspect
the alignment space. The projected arrays do not contain raw or
participant-level EEG and are not required by `train_ext.py`.

Projector checksums:

| File | SHA-256 |
| --- | --- |
| `projector_image.pt` | `ff1d9a99fee6eb36c050774cb0c9bdf8796d07649d646a6bdbfb18d5d64f715d` |
| `projector_text.pt` | `f5c7215b9147c93559776f2560d248d21c5a1a8bdc5bc26b3a5ccd68ef83d55b` |

To retrain the projectors, configure the feature paths in
`train_text_image_alignment.py`. Generated projectors should be published with
the source commit, configuration, input-feature checksums, and evaluation
notes.
