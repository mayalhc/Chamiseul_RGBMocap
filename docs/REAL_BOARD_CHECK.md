# Checking a real printed reference board

The synthetic checks composite the sheet into footage, so the sheet's own size is
assumed. A real sheet adds two facts that only a tape measure can supply: how big
the print actually is, and how high the camera actually was. This procedure uses
those two numbers to verify the measured floor against the physical room.

## 1. Print the sheet

1. Motion Correction → **Flat Ground** → tick **Measure Floor (Reference Board)**.
2. Choose the size. A full-body shot from 2–4 m needs **A2 or A1**; at A4 each
   marker shrinks below the 16 px it needs and the sheet is rejected. Smaller
   sheets only work when the camera is close.
3. Click **Printable Reference Sheet...** and print the PNG at **100% / actual
   size**. Turn off "fit to page", "shrink to fit" and "scale to printable area".
4. Measure the printed 100 mm ruler with a ruler or tape and write the number
   down. If it is not 100 mm the print was scaled, and every measured height will
   be scaled by 100/ruler.

The app reads the printed size from the corner marker ids, so you do not have to
tell it which size you printed.

## 2. Set up and measure

1. Lay the sheet **flat on the floor where the actor will stand** and tape the
   corners down. A curled sheet is no longer a plane, and the measurement will
   fail or drift.
2. Fix the camera so the whole body and all four markers are in frame for a few
   seconds at the start of the clip.
3. Measure with a tape:
   - **Lens height**: floor to the centre of the lens, perpendicular. This equals
     the distance the detector reports between the lens and the sheet's plane.
   - **Ruler**: the printed 100 mm ruler (step 1).
   - Optional: floor distance from the lens to the sheet centre, to sanity-check
     the recovered depth.
4. Have the actor stand on or next to the sheet for at least a second, then
   perform the motion. Staying near the sheet keeps the single floor height
   valid.

## 3. Reconstruct, then check

Process the clip in the app as usual (import, analyze). The check needs the
camera intrinsics the reconstruction produces.

```
runtime\python.exe -B tools\verify-floor-board-real.py ^
    projects\<name>_project\<model> ^
    --camera-height 1243 --ruler-mm 100
```

Without `--camera-height` / `--ruler-mm` the tool still prints every measured
number so you can compare it with your tape by hand. The report is written to
`screenshots/floor-board-real.json` unless `--json` says otherwise, and the exit
code is 0 only when every supplied measurement passes.

## 4. Reading the report

| Field | Meaning | Pass |
| --- | --- | --- |
| `lens_height.measured_mm` | Lens to sheet-plane distance solved from the markers | compare with tape |
| `lens_height.delta_mm` | Difference from your tape | within 15 mm or 1% |
| `print.scale` | Longest marker-corner distance, measured / printed | within 0.5% of `100 / ruler` |
| `agreement_mm` | Spread of the plane across sampled frames | under ~30 mm |
| `marker_px` / `span_px` | Marker edge and sheet footprint in pixels | marker 16 px or more |
| `tilt.deg`, `tilt.max_mm_per_m` | Floor tilt in the app's vertical frame and its worst-case drift per metre | informational |
| `grounding.floor_mm` | Floor the correction actually used, at the reported anchor | equals the measured plane there |
| `grounding.sole_height_mm` | Corrected sole height above that floor | under 10 mm |
| `grounding.actor_to_sheet_m` | How far the actor stood from the sheet | keep small |
| `grounding.tilt_cost_mm` | Height the single floor value spent crossing that distance | small next to `max_mm_per_m × distance` |

## 5. When it fails

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `detected: false` | Markers too small, blurred, glare, or partly out of frame | print A2/A1, move closer, diffuse the light, keep the sheet still |
| `lens_height` off by more than a few cm | tape measured to a different point, sheet not flat, or intrinsics wrong (cropped/resized/zoomed clip) | re-measure, flatten the sheet, re-record without digital zoom |
| `print` fails | printer scaled the page | reprint at 100%, measure the ruler again |
| `agreement_mm` large | motion blur on the corners | hold the camera still, better light |
| `tilt_cost_mm` grows with take length | actor walked far from the sheet on a tilted floor | keep the actor near the sheet, or use Multi Level for slopes |

## 6. Confirm grounding

Tick **Measure Floor (Reference Board)**, set **Correction Mode → Flat Ground**
and click **Apply Correction**. The correction report should carry a
`floor_board` block with the measured plane and the anchor it was read at, and
the soles should sit on that floor.

## Limits

- A project must already be reconstructed: the sheet is solved with the take's
  own intrinsics.
- The sheet's plane is the floor's plane only while the sheet lies flat, face up.
- This validates the floor's metric position and scale. It does not validate
  contact timing or how well the motion itself matches the body.
- There is no gravity reference. The app's vertical axis comes from the camera,
  and the floor is measured relative to that axis; a camera held crooked yields a
  floor tilted with it, which the report shows as `tilt`.
