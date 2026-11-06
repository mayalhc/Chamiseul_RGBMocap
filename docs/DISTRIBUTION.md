# Publishing RGB MoCap through Git

RGB MoCap ships as source plus pinned installation metadata. Git carries the
application, the upstream model source and the installer locks; each PC builds
its own runtimes with `install.cmd`, and model weights stay with their providers.

Numbers below were measured in this checkout on 2026-10-28 with about 2,460
tracked files and an installed `runtime*`, `models` and `projects` tree.

## What the repository carries

| Tracked path | Contents | Size |
| --- | --- | --- |
| `mocap/`, `app.py`, `launch_source.py`, `worker_boot.py`, `scene_boot.py` | application source | small |
| `install.cmd`, `update.cmd`, `RGBMoCap.cmd` | entry points users run | small |
| `tools/` | bootstrap, installer, preflight, source check | small |
| `installer/` | `runtime-lock.json`, `git-support-lock.json`, `external-assets.json`, pure-Python wheel sources, licenses | 2.6 MiB |
| `third_party/` | pinned upstream model source (2,104 of the tracked files) | ~25 MiB |
| `prebuilt/motion-correction/_motion_correction.cp310-win_amd64.pyd` | verified CPU correction module, hash pinned in `git-support-lock.json` | 404 KiB |
| `native_compat/`, `assets/`, `docs/` | shims, fonts/icon, documentation, two sample videos | ~36 MiB |
| `requirements/`, `requirements-*.txt` | runtime pins and worker cache fingerprints | small |

Whole checkout: **about 2,460 files, 62 MiB**, dominated by `docs/samples`
(25 MiB of sample video) and `assets/fonts/NotoSansKR.ttf` (9.9 MiB).

`tools/check_source.py` is the gate for this composition: it parses every
tracked Python file, rejects generated or model payloads, rejects extra AI
instruction files and verifies the recorded auxiliary data hashes.

## What `install.cmd` downloads on the user's PC

| Item | Size | Where |
| --- | --- | --- |
| Six pinned runtime profiles (7.6 GB of wheels) | installed estimate 12.5 GiB | `.setup/downloads`, then `runtime*` |
| Blender 4.5.0 | 400 MB | `runtime_character` |
| FFmpeg 7.1.1 | 178 MB | `tools/bin` |
| Upstream support files | small | pinned in `installer/external-assets.json` |

First installation needs **32.4 GiB free** including temporary space. These
files are excluded from Git because they are platform-specific, large and
carry their own licenses; `installer/runtime-lock.json` and
`installer/git-support-lock.json` pin every URL, byte count and SHA-256 so each
PC installs identical bytes.

## What users must supply

About **29 GiB of model weights and registered body models**. They are never in
Git and never redistributed by the app, because the providers require
registration, license acceptance or both. The model-by-model table, exact file
names and destination paths are in
[Model Manager paths](USER_GUIDE.md#where-each-model-file-goes).

A second PC can reuse an existing library: close both apps and copy the whole
`models` folder next to the new `RGBMoCap.cmd`, then confirm each row in Model
Manager. See [Reuse models in a new installation](USER_GUIDE.md#reuse-models-in-a-new-installation).

## Never commit

All of these are ignored already; the point of the list is to keep them ignored
when adding new tools or models.

| Path | Why | Size here |
| --- | --- | --- |
| `runtime*/` | installed per PC by `install.cmd`, checked by hash | 12 GiB today |
| `models/` | license-restricted weights | 18 GiB |
| `projects/`, `character_scenes/` | user work | 2.2 GB |
| `cache/` | re-creatable Hugging Face, Torch and ExMR caches | 12 GiB |
| `third_party/*/inputs/`, `third_party/*/checkpoints/`, `third_party/*/ckpts/` | local model inputs for FootMR, HTD-Refine, GVHMR and FlowHMR | 29 GiB |
| `.setup/`, `SETUP_RECEIPT.json` | per-machine installation state | — |
| `*.pyd`, `*.dll`, `*.exe`, `*.whl`, `*.zip` except the pinned prebuilt | build products | — |
| `.env*`, `*.pem`, `*.key` | credentials | — |

The model-input folders were a live hazard before this change: FootMR and
HTD-Refine inputs were untracked but **not ignored**, so one `git add -A` would
have staged 29 GiB of weights. Keep the rule when adding another model:

```gitignore
/third_party/<model>/inputs/
/third_party/<model>/checkpoints/
```

## Publish checklist

Run from the repository root with the installed runtime:

```cmd
runtime\python.exe -B tools\check_source.py
runtime\python.exe -I -B tools\preflight.py
runtime\python.exe -I -B -m compileall -q mocap tools app.py launch_source.py worker_boot.py
git status --short
```

1. `check_source.py` must print `"errors": []`.
2. `preflight.py` must end with `Result OK` on the publishing PC.
3. `git status --short` must list only intended edits: no `runtime*`, `models`,
   `projects`, `cache` or `third_party/*/inputs` entries.
4. If application source changed, refresh the recorded `SOURCE_MANIFEST.json`
   snapshot through the release process used for the previous publish; no
   tracked tool regenerates it.
5. Commit and push `main`. Installed copies pick the change up with
   `update.cmd`, which refuses local edits, requires `main` and fast-forwards.

Dependency changes need one more step. The launcher compares SHA-256 values
recorded at install time for `installer/runtime-lock.json`,
`mocap/runtime_registry.py`, `installer/git-support-lock.json` and
`installer/external-assets.json`; changing any of them makes every installed
copy demand `install.cmd` before it starts. That is intentional, but update the
files together:

- Profile pins live in `mocap/runtime_registry.py`; `runtime-lock.json` stores
  `registry_sha256` for it. `tools/runtime_support.py` refuses to prepare a lock
  while the two differ, so update the lock deliberately and rerun the model
  checks for every profile you touched.
- Wheels, versions or URLs change `runtime-lock.json`, which also changes its
  `id`; installed PCs then perform a full verified install instead of reusing
  the old receipt.

## Install on another PC

1. Install [Git for Windows](https://git-scm.com/download/win) and clone into a
   short path such as `D:\RGBMoCap`:

   ```cmd
   git clone https://github.com/mayalhc/Chamiseul_RGBMocap.git RGBMoCap
   cd RGBMoCap
   ```

   Path length matters: the staged installation reaches about 216 characters
   below the checkout folder, so a deep `C:\Users\<name>\Documents\...` folder
   can exceed the 260-character Windows limit. `preflight.py` reports this.

2. Run `install.cmd`. It now prints an installation check before downloading:
   Windows version, GPU and compute capability, driver, VRAM, path length and
   free disk. It stops with a clear message when the PC cannot run the pinned
   kernels, instead of failing after the download.

3. Prepare model files in **Model Manager**, or copy a `models` folder from an
   existing installation.

4. Run `RGBMoCap.cmd`. Use `update.cmd` for later source updates and
   **Help → Check System** to re-check the hardware after a driver change.

## GPU support

The pinned wheels are CUDA 12.8 builds of Torch 2.7.1 (`shared`, `gui`, `flow`,
`hand`, `htd`) and 2.8.0 (`exmr`). Their compiled kernels cover compute
capability 5.0, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 9.0, 10.0 and 12.0.

| Series | Capability | Supported | Minimum Windows driver |
| --- | --- | --- | --- |
| RTX 20 (Turing) | 7.5 | yes, kernels compiled for 7.5 | 527.41 |
| RTX 30 (Ampere) | 8.6 | yes | 527.41 |
| RTX 40 (Ada) | 8.9 | yes, runs the 8.6 kernels (same major version) | 527.41 |
| RTX 50 (Blackwell) | 12.0 | yes, kernels compiled for 12.0 | 570 or newer |

`mocap/gpu_support.py` holds this rule set, `tools/preflight.py` applies it
before installation and `Help → Check System` applies it to the installed
build. Driver and capability are checked separately, so an unplugged GPU, a
30-series card on a 2021 driver and a 50-series card on a 546 driver each get a
distinct message.

VRAM is not a hard gate. Measured peaks are 11.3 GiB (ExMR), 14.9 GiB
(FlowHMR) and 5.1 GiB (GVHMR); the check warns below 12 GiB and again below
6 GiB, where only short clips in **Low Memory** mode are realistic. Cards with
6–11 GiB therefore install and run, with fewer frames per take.

`install.cmd` blocks only on a genuinely unsupported combination. To install
on such a PC for inspection, set `RGBMOCAP_ALLOW_UNSUPPORTED_GPU=1` before
running it.

## Repositories

- Distribution checkout: `https://github.com/mayalhc/Chamiseul_RGBMocap`
  (this fork contains the current retargeting, export and hardware checks).
- Upstream source: `https://github.com/toyxyz/toyxyz_RGBMocap` (distribution checkout it forks: `https://github.com/creatorchamiseul/toyxyz_RGBMocap`).

Documentation clone URLs must match the repository users are told to clone;
cloning upstream gives an older build without these changes.
