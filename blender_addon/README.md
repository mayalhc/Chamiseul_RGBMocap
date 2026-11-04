# RGB MoCap Bridge (Blender extension)

The parts of RGB MoCap that make sense inside Blender, without the app, the AI
models or its Python runtimes:

- **Humanoid rig detection** — match a rig against the canonical 22 Mixamo body
  bones using the same alias rules as the app (Mixamo, Unreal, glTF/GLB, VRM,
  Rigify-style limb names, `DEF-`/`ORG-`/`MCH-` prefixes).
- **Canonical renaming** — rename the mapped bones and their skin groups to the
  canonical names, so an Unreal or Rigify rig becomes retargetable/importable
  where Mixamo names are expected. Unmapped bones (fingers, helpers) keep their
  names.
- **Flat-ground snapping** — plant the skinned character on one floor plane.
  `Plant feet` corrects every frame, `Drop` applies one offset from the lowest
  frame, `Auto drop` uses a low percentile so deliberately airborne frames are
  not pulled down.
- **FBX / GLB export** — bake the bone animation and write one FBX (with or
  without the skinned mesh) or one binary glTF, using the export options from
  the app's character export.

What it deliberately does not do: video motion capture, AI inference,
retargeting from a source person, or the app's private FBX sampler that forces
rest-pose bone defaults and body-only bone sampling. Export from the addon is
Blender's native exporter with the app's option set; use the app when you need
its rest-pose/bind-pose guarantees.

## Install

This is a Blender 4.2+ extension (`blender_manifest.toml`, id
`rgbmocap_bridge`). Build and install it into the Blender you use with one
command:

```cmd
python blender_addon\install.py --blender "C:\Program Files\Blender\Blender 5.2\blender.exe"
```

Without `--blender` it uses `BLENDER_EXE` or the newest Blender it can find. It
builds `blender_addon/dist/rgbmocap_bridge-1.0.0.zip` with Blender's own
`--command extension build`, removes any older legacy `scripts/addons` copy so
the UI is not registered twice, then installs and enables the package in the
`user_default` repository. Blender reports it as
`bl_ext.user_default.rgbmocap_bridge`.

Equivalent manual steps:

```cmd
blender.exe -b --command extension build --source-dir blender_addon\rgbmocap_bridge --output-dir blender_addon\dist
blender.exe -b --command extension install-file -r user_default -e blender_addon\dist\rgbmocap_bridge-1.0.0.zip
```

Build only (for publishing or handing the zip to someone else) with
`--build-only`. To update, rerun the installer; to remove, use Blender's
**Preferences > Get Extensions > Installed**.

### Several Blender versions on one PC

Blender 4.x/5.x builds on one machine share the extension repository folder
`%APPDATA%\Blender Foundation\Blender\5.0\extensions\user_default`, so the
package only needs to be installed there once. Enabling is still per version
(each Blender keeps its own preferences), so run the installer once with each
Blender you use, or enable **RGB MoCap Bridge** once under **Preferences > Get
Extensions > Installed** in that version. `install.py` prints the resulting
`bl_ext.user_default.rgbmocap_bridge` state.

## Live link

**File > Live Link to Blender** in the app streams the pose on screen to this
extension, with no export step and no AI rerun.

1. In the app, open the project and turn on **File > Live Link to Blender**
   (listens on `127.0.0.1:9187`; `RGB_MOCAP_LIVE_PORT` changes it).
2. In Blender, **RGB MoCap > Live link > Start Live Link** (port must match).
3. The character the app is showing appears by itself; play, pause or scrub in
   the app and that same character follows, at up to 60 updates per second.

**Nothing needs importing, and there is no stand-in mesh.** The app sends its own
base body — the SMPL surface it draws — or the character you imported there, as a
rest file plus one matrix per bone per frame. **Load Base Character** brings that
same character in by hand, so Blender holds exactly what the app holds.

When a character is on screen the app sends **that character first**: its own
rest file, bone palette and scene placement. Blender appends the same rest file,
so the rig, skin, textures and rest pose are the app's own, and then applies one
matrix per palette bone per frame — bone positions and rotations are the app's
values, with no guessed axes or retargeting. The status line shows the frame that
is actually posed. That includes the app's base body: the SMPL-X surface it draws
is sent as its own rest file, so Blender shows the same body, not a proxy.

Only when the app is showing a video rather than a take does it fall back to the
22 canonical body joints, which aim an armature you imported and mapped at
streamed segment directions; with nothing in the scene Blender waits and says so
instead of inventing a figure. That joint fallback is a preview, since bone roll
and IK are not solved.

### Recording live poses

**Record Live to Action** keys every incoming pose into a new action named
`RGBMoCap Live`, starting at the current scene frame, and **Stop Recording**
finishes it:

- One key per mapped bone (rotation, plus the hips position) on every incoming
  app frame; app frame numbers map to scene frames one to one, so playback speed
  matches the source.
- Quaternion sign continuity is enforced, keys use linear interpolation, and the
  scene range and frame rate are set to the recording, so scrubbing or playing
  the action reproduces the live motion.
- The previous action of that armature is detached when recording starts; keep
  a copy first if you need it.

## Use

1. Import a character (**File > Import**, or the addon's **Import Character**
   button) and select its armature.
2. **3D View > Sidebar (N) > RGB MoCap**.
3. **Scan Humanoid Rig** shows which body bones were matched and what is
   missing. **Use Canonical Names** renames a complete rig.
4. Set the floor height (metres), choose a mode and press **Snap To Floor**. The
   panel reports the lowest point before and after, so contact error is visible
   rather than assumed.
5. **FBX** / **GLB** export the selected armature and its skinned meshes.

Snapping writes one location key per frame on the root bone (`Hips` when
present), so the correction survives export; undo with Ctrl+Z.

## Check

`tools/verify-blender-addon.py` runs the whole thing headless on a real Mixamo
character (import, detect, rename, float the clip, plant it, export and
re-import FBX and GLB). It uses the installed extension when Blender has it
enabled, and the checkout otherwise:

```cmd
blender.exe --factory-startup -b --python tools\verify-blender-addon.py
```

It writes `screenshots/blender-addon-report.json` and exits non-zero when a
check fails. Set `RGBMOCAP_ADDON_REPORT` to write the report somewhere else,
e.g. when checking another Blender version.

The live link has these checks:

- `tools/check-live-link.py` — wire format, run with the app runtime.
- `tools/verify-live-base-model.py` — with nothing imported, Start Live Link
creates the base humanoid, sizes it from the stream and follows every joint
(currently 0.02° direction and 0 mm hip error; the panel button is checked too).
- `tools/verify-live-character.py` — the character stream end to end in Blender:
the character is appended at rest before any pose arrives, then every posed bone
is compared with the take's own exported animation (currently 0.03° and 0.001 mm).
- `tools/verify-live-character-app.py` — the real app window describes the
character on screen before any pose packet.
- `tools/verify-live-app.py` — the real app window sends the joint stream it is
showing, exactly.
- `tools/verify-live-link.py` — Blender plays a moving joint stream on a rig.
- `tools/check-addon-icons.py` — every icon the panel asks for exists in the
Blender that will draw it (Blender drops icon names between versions).

After changing the addon, rerun `blender_addon/install.py`, then **Reload
Scripts** in Blender (or restart it) so the running session picks up the new
module.
