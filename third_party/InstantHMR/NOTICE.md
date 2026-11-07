# InstantHMR (vendored subset)

Source: https://github.com/Naturalpad-studio/InstantHMR, commit
`5437400d4b25c1972821e51916fac34bd96acce6`. Code license: Apache-2.0 (see `LICENSE`).

Included: `instanthmr/inference.py` and `instanthmr/models.py` (unmodified) and a
reduced `instanthmr/__init__.py`. Not included: the demo, detector, visualiser,
training and benchmark code, and all weights.

Weights are not bundled. RGB MoCap's Model Manager downloads them from
https://huggingface.co/momolesang/InstantHMR at a pinned revision, with SHA-256 checks:

- `nano`, `tiny`, `small` and `medium`: CC BY-NC 4.0, non-commercial use only.
- `medium-commercial`: Apache-2.0, trained only on images that allow commercial use.
- `mhr-body.onnx` (Momentum Human Rig body graph): Apache-2.0.

Paper: Adjel, Chalabi, Bonnet, "InstantHMR: Real-Time Human Mesh Recovery on Your Phone",
HAL preprint hal-05854387 (2026).
