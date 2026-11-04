"""Check a physically printed floor reference board against a tape measure.

Every synthetic check composites the sheet into footage, so the sheet's own
metric size is assumed, not observed. This tool closes that gap. A sheet lying
flat on the floor makes two of its properties measurable with a tape, with no
help from the reconstruction:

  * the perpendicular distance from the lens to that floor plane, which is
    exactly what detection reports as the measured camera height, and
  * the printed size, checked by measuring the 100 mm ruler printed on the sheet.

Both are lengths in metres, so agreement means the intrinsics, the marker
geometry and the printed sheet all agree with the physical world. Nothing here
uses foot contacts, so the floor it validates is independent of the motion.

Run from the app directory, on a take that has already been reconstructed:

  runtime/python.exe -B tools/verify-floor-board-real.py <project> \
      --camera-height 1243 --ruler-mm 100 --json build/floor-board-real.json

Only the first argument is required. Without measurements the tool still prints
every number next to what it should be compared with.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
BASE = APP.parent
sys.path.insert(0, str(APP))

# A tape measure on a tripod is good to a few millimetres; 15 mm also covers a
# lens that is not exactly above the point the user measured from.
HEIGHT_TOLERANCE_MM = 15.0
HEIGHT_TOLERANCE_FRACTION = 0.01
# Printers that "fit to page" shrink silently. The ruler makes that visible, and
# a 0.5% scale error already moves a 1.2 m floor by 6 mm.
PRINT_TOLERANCE = 0.005
# A sole this far above or below the assigned floor is still fine: the contact
# model works in millimetres and floor contact is not exact.
SOLE_TOLERANCE_MM = 10.0


def correction_summary(project, kind):
    """Registration numbers from an existing correction run, if any."""
    report_path = Path(project) / f'{kind}_correction.json'
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding='utf-8'))
    board = report.get('floor_board') or {}
    return dict(kind=kind,
                floor_mm=None if report.get('floor') is None else round(float(report['floor']) * 1000, 2),
                board_used=bool(board),
                board_anchor_mm=board.get('anchor'),
                sole_height_mm=None if report.get('contact_sole_height_after_m') is None else
                round(float(report['contact_sole_height_after_m']) * 1000, 3),
                sole_below_floor_mm=None if report.get('sole_below_floor_after_m') is None else
                round(float(report['sole_below_floor_after_m']) * 1000, 3),
                penetration_over_1cm=report.get('contact_sole_penetration_over_1cm'),
                warning=report.get('warning') or '')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate a printed floor reference board.')
    parser.add_argument('project', help='reconstructed project folder containing input.mp4')
    parser.add_argument('--camera-height', type=float, default=None,
                        help='tape-measured lens height above the floor, in millimetres')
    parser.add_argument('--ruler-mm', type=float, default=None,
                        help='measured length of the printed 100 mm ruler, in millimetres')
    parser.add_argument('--kind', default='corrected_footmr_initial',
                        help='correction result to summarise when one exists')
    parser.add_argument('--json', default=str(APP / 'screenshots/floor-board-real.json'),
                        help='where to write the report')
    args = parser.parse_args(argv)

    project = Path(args.project).resolve()
    if not (project / 'project.json').is_file():
        raise SystemExit('Not a project folder: ' + str(project))
    from mocap.floor_board import detect_for, floor_for

    report = dict(project=str(project))
    record = detect_for(project, verbose=lambda line: print(line))
    report['detected'] = record is not None
    if record is None:
        report['ok'] = False
        report['advice'] = ('No readable sheet. Print the sheet at 100% (actual size) with '
                            'A2 or A1 for a full-body shot, lay it flat where the actor stands, '
                            'and keep all four markers visible and sharp for a few seconds.')
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return 1

    measured_mm = float(record['plane']['offset']) * 1000
    print_scale = float(record['print_scale'])
    report.update(paper=record['paper'], markers=record['markers'], frames=len(record['frames']),
                  marker_px=round(record['marker_px'], 1), span_px=round(record['span_px'], 1),
                  agreement_mm=round(record['agreement_mm'], 2),
                  reprojection_px=round(record['reprojection_px'], 2),
                  plane_rms_mm=round(record['rms_mm'], 2))
    report['lens_height'] = dict(measured_mm=round(measured_mm, 1), tape_mm=args.camera_height)
    report['print'] = dict(measured_span_mm=round(float(record['corners_mm']), 1),
                           expected_span_mm=round(float(record['corners_truth_mm']), 1),
                           scale=round(print_scale, 5), ruler_mm=args.ruler_mm)

    verdict = dict(detected=True)
    if args.camera_height is not None:
        delta = measured_mm - float(args.camera_height)
        limit = max(HEIGHT_TOLERANCE_MM, HEIGHT_TOLERANCE_FRACTION * float(args.camera_height))
        report['lens_height'].update(delta_mm=round(delta, 1), tolerance_mm=round(limit, 1))
        verdict['lens_height'] = abs(delta) <= limit
    if args.ruler_mm is not None:
        # A sheet printed small carries smaller markers, so the solver assumes
        # they are farther away and reports distances larger by 100/ruler. The
        # ruler is the physical proof of that factor.
        target = 100.0 / float(args.ruler_mm)
        delta = print_scale - target
        report['print'].update(ruler_scale=round(target, 5), delta=round(delta, 5),
                               tolerance=PRINT_TOLERANCE)
        verdict['print'] = abs(delta) <= PRINT_TOLERANCE
        if not verdict['print']:
            report['advice'] = ('The printed scale and the measured scale disagree. Reprint at '
                                '100% / actual size (no fit to page) and measure the printed ruler '
                                'again before trusting the floor height.')

    floor_value, detail = floor_for(project)
    if floor_value is not None:
        plane = detail['plane']
        report['floor'] = dict(height_mm=round(float(floor_value) * 1000, 2), paper=detail['paper'],
                               anchor_mm=[round(v, 4) for v in detail['anchor']],
                               sheet_centre_mm=[round(v, 4) for v in detail['board_center']])
        # One floor height cannot follow a tilted floor. This is the worst-case
        # drift a metre away, in the app's own vertical axis.
        slope = math.tan(math.radians(float(plane['tilt_deg'])))
        report['tilt'] = dict(deg=round(float(plane['tilt_deg']), 3), max_mm_per_m=round(slope * 1000, 2))

        grounding = correction_summary(project, args.kind)
        if grounding is not None:
            report['grounding'] = grounding
            sheet_centre = np.asarray(detail['board_center'], np.float64)
            if grounding['sole_height_mm'] is not None:
                grounding['sole_ok'] = abs(grounding['sole_height_mm']) <= SOLE_TOLERANCE_MM
            if grounding['board_anchor_mm']:
                anchor = np.asarray(grounding['board_anchor_mm'], np.float64)
                travel = float(np.hypot(*(anchor[[0, 2]] - sheet_centre[[0, 2]])))
                grounding['actor_to_sheet_m'] = round(travel, 2)
                if grounding['floor_mm'] is not None:
                    # How much of the single floor height was spent covering the
                    # distance between the sheet and where the actor stood.
                    grounding['tilt_cost_mm'] = round(grounding['floor_mm'] - report['floor']['height_mm'], 2)

    report['verdict'] = verdict
    report['ok'] = bool(all(verdict.values()))
    text = json.dumps(report, indent=2)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(text, encoding='utf-8')
    print(text)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
