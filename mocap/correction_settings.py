"""Lightweight correction settings shared by GUI, cache and CPU worker."""

PRESETS = {
    'weak': dict(confidence=.75, release=.75, minimum=.06, ramp=.16,
                 anchor_drift=.12, strength=.6, gap=0.),
    # Keep legacy contact thresholds, timing and strength. Solver defect fixes
    # are shared by all presets and may change the corrected pose.
    'normal': dict(confidence=.65, release=.65, minimum=.06, ramp=.1,
                   anchor_drift=.12, strength=1., gap=0.),
    'strong': dict(confidence=.65, release=.5, minimum=.06, ramp=.06,
                   anchor_drift=.20, strength=1., gap=.08),
}

DESCRIPTIONS = {
    'weak': 'Gentler locking; keeps more original foot movement. Contact position correction is 60%, with smooth transitions outside contacts.',
    'normal': 'Balanced foot locking. Full position holds in stable contacts, with adaptive transitions and smooth automatic travel handoffs. Inferred foot rotation is retained.',
    'strong': 'Firmer holds: keeps lower-confidence contacts and bridges brief grounded dropouts. Stable contacts use full position holds; automatic travel handoffs remain smooth. Avoid deliberate foot slides; inferred foot rotation is retained.',
}

MODES = {'flat': 'Flat Ground', 'multi_level': 'Multi Level'}
MODE_HELP = ('Flat Ground uses one correction floor. Multi Level keeps source elevation and locks each stance at its own 3D position. '
             'Use for stairs, slopes and climbing. It preserves inferred distal rotations and does not reconstruct terrain. Overhead hand contacts require manual timing.')


def correction_options(options=None):
    import math
    options = options or {}
    mode=options.get('mode','flat')
    if mode not in MODES:raise ValueError('Unknown Motion Correction mode: '+str(mode))
    preset = options.get('preset', 'normal')
    if preset not in PRESETS:
        raise ValueError('Unknown Motion Correction preset: ' + str(preset))
    auto = bool(options.get('auto_floor', False))
    floor = 0. if auto else float(options.get('floor', 0.))
    if not math.isfinite(floor):
        raise ValueError('Correction floor must be finite')
    pelvis = float(options.get('pelvis_offset', 0.))
    if not math.isfinite(pelvis) or abs(pelvis) > .25:
        raise ValueError('Pelvis height offset must be between -0.25 and +0.25 meters')
    result={'auto_floor': auto, 'floor': floor, 'preset': preset, 'pelvis_offset': pelvis, 'mode':mode, 'foot_lock':bool(options.get('foot_lock',True)), 'hand_lock':bool(options.get('hand_lock',False))}
    # Only an enabled board is recorded, so existing receipts and signatures of
    # projects that never used one stay valid.
    if options.get('floor_board'):result['floor_board']=True
    if options.get('occlusion_fill', True):result['occlusion_fill']=True
    return result


def effective_options(project,state):
    """Per-actor locking preset, but a people group's floor is authoritative."""
    options=correction_options(state.get('correction',{}))
    from .scene_floor import scene_root,read_config
    root=scene_root(project,state)
    shared=read_config(root) if root is not None else None
    return {**options,**shared,'mode':shared.get('mode','flat')} if shared is not None else options
