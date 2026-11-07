# InstantHMR: fast live estimator

[Original project](https://github.com/Naturalpad-studio/InstantHMR) · [Weights](https://huggingface.co/momolesang/InstantHMR) ·
[Paper and BibTeX](../CITATIONS.md#instanthmr)

InstantHMR is an optional estimator for the **live character** (webcam or web video link). It is not a
reconstruction model: it does not create a project, use Find person for offline takes, or replace
GVHMR, ExMR or FlowHMR. It gives the same 70 body keypoints in camera space (the Momentum Human Rig
layout) that SAM 3D Body gives the live solver, so the rest of the live path is unchanged: the app's own
person box, smoothing, the rig solve, the 3D view, and Live Link to Blender or Cascadeur.

## Install

1. Close RGB MoCap and run `Setup_InstantHMR.cmd`. It adds ONNX Runtime GPU (about 0.4 GiB, pinned by hash) in
   `runtime_instanthmr`; no existing runtime is changed. Python, NumPy and the CUDA libraries come from the ExMR runtime.
2. Choose **File > Live Body Estimator > InstantHMR ...**. Missing model files (about 142 MiB for medium) download
   from Hugging Face at a pinned revision with SHA-256 checks. No sign-in is needed.
3. The choice applies the next time the live character starts. Until the files are in place, SAM 3D Body is used.

| Choice | Weights | License |
| --- | --- | --- |
| InstantHMR medium (fast) | `instanthmr-medium.onnx` | CC BY-NC 4.0, non-commercial |
| InstantHMR nano (fastest) | `instanthmr-nano.onnx` | CC BY-NC 4.0, non-commercial |
| InstantHMR medium-commercial | `instanthmr-medium-commercial.onnx` | Apache-2.0 |

All three use `mhr-body.onnx` (Apache-2.0). The code is Apache-2.0 and vendored in `third_party/InstantHMR`
(see its NOTICE.md). Whether a weight license permits your use is your responsibility; see
[third-party notices](../../THIRD_PARTY_NOTICES.md).

## Measured here

One RTX 5080, a 7 s 1080p single-person clip, the app's own live path. These are measurements on one PC, not guarantees.

| Estimator | Estimator time per frame | Live frames per second (20 s window) |
| --- | --- | --- |
| SAM 3D Body | about 115-120 ms | about 8 |
| InstantHMR medium | about 18 ms | about 21 |

- Estimator time includes cropping, the network and the body model. The whole worker step was about 23 ms
  per frame with InstantHMR, so its capacity is above the 21 fps seen; the app's own timers and the playing clip bound the live rate.
- Against SAM 3D Body on the same 30 frames, pelvis-relative joint positions differed by 5.1 cm (medium) and
  8.4 cm (nano) on average. SAM 3D Body is the reference here, not ground truth. Absolute depth differed by
  about 45 cm for a person roughly 13 m from the camera.
- nano was no faster than medium on this PC (about 17 ms either way) and less accurate.
- Not measured: webcam input on real hardware, several people, GPUs smaller than 16 GB, and
  accuracy against ground truth.

## Limits

- It estimates each frame on its own; smoothing is the live solver's. There is no world-grounded or
  camera-motion correction. Depth uses the same assumed focal length as the SAM live path.
- It tracks one performer, like the existing live character.
- Lower accuracy than SAM 3D Body is expected; use SAM 3D Body when quality matters more than latency.
