# Copyright (c) 2026 Chamiseul
# SPDX-License-Identifier: GPL-3.0-or-later (see blender_addon/LICENSE)
"""Humanoid bone detection for imported character rigs.

The canonical names are the Mixamo names RGB MoCap already expects, and the
alias table mirrors the app's retargeting rules so a rig that the app can
retarget is also recognised here. Detection never touches the scene; renaming
is a separate, explicit step.
"""
from . import NAMES

# Mixamo canonical names are matched by their normalized form (prefix and
# separators removed), then the exporter aliases follow.
GLB_ALIASES = {
    'Hips': ('hips', 'pelvis', 'bip01pelvis', 'hip'),
    'Spine': ('spine', 'spine01', 'spine0', 'abdomen'),
    'Spine1': ('spine1', 'spine02', 'spine001', 'chest'),
    'Spine2': ('spine2', 'spine03', 'spine3', 'spine002', 'upperchest'),
    'Neck': ('neck', 'neck01'),
    'Head': ('head', 'head01'),
    'LeftShoulder': ('leftshoulder', 'shoulderl', 'claviclel', 'lclavicle'),
    'RightShoulder': ('rightshoulder', 'shoulderr', 'clavicler', 'rclavicle'),
    'LeftArm': ('leftarm', 'leftupperarm', 'upperarml', 'lupperarm', 'arml'),
    'RightArm': ('rightarm', 'rightupperarm', 'upperarmr', 'rupperarm', 'armr'),
    'LeftForeArm': ('leftforearm', 'leftlowerarm', 'lowerarml', 'forearml', 'lelbow'),
    'RightForeArm': ('rightforearm', 'rightlowerarm', 'lowerarmr', 'forearmr', 'relbow'),
    'LeftHand': ('lefthand', 'handl', 'lhand', 'wristl'),
    'RightHand': ('righthand', 'handr', 'rhand', 'wristr'),
    'LeftUpLeg': ('leftupleg', 'leftupperleg', 'thighl', 'upperlegl', 'lthigh'),
    'RightUpLeg': ('rightupleg', 'rightupperleg', 'thighr', 'upperlegr', 'rthigh'),
    'LeftLeg': ('leftleg', 'leftlowerleg', 'calfl', 'lowerlegl', 'shinl', 'lshin'),
    'RightLeg': ('rightleg', 'rightlowerleg', 'calfr', 'lowerlegr', 'shinr', 'rshin'),
    'LeftFoot': ('leftfoot', 'footl', 'lfoot'),
    'RightFoot': ('rightfoot', 'footr', 'rfoot'),
    'LeftToeBase': ('lefttoebase', 'lefttoes', 'lefttoe', 'toebasel', 'balll', 'ltoe', 'toel'),
    'RightToeBase': ('righttoebase', 'righttoes', 'righttoe', 'toebaser', 'ballr', 'rtoe', 'toer'),
}

# Exported rigs often carry an origin namespace such as DEF-, ORG-, MCH-.
PREFIXES = ('mixamorig', 'def', 'org', 'mch')
FOOT_BONES = ('LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase')


def normalized(name):
    """Drop exporter namespaces and separators, e.g. 'mixamorig:LeftUpLeg'."""
    tail = name.replace('|', ':').split(':')[-1]
    lowered = ''.join(ch for ch in tail.lower() if ch.isalnum())
    for prefix in PREFIXES:
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            return lowered[len(prefix):]
    return lowered


def _candidates(bones):
    lookup = {}
    for name in bones:
        lookup.setdefault(normalized(name), []).append(name)
    return lookup


def _match(lookup, aliases, canonical):
    """Exact canonical name first, then aliases in their declared order."""
    for alias in (normalized(canonical),) + tuple(aliases):
        hits = lookup.get(alias, ())
        if len(hits) == 1:
            return hits[0]
        if hits:
            # Two bones share a name after normalization (both hands of a
            # mirrored helper rig); prefer the shortest, keep it deterministic.
            return sorted(hits, key=lambda value: (len(value), value))[0]
    return None


def _rigify_spine(bones, found):
    """Rigify names the hips bone 'spine' and continues with 'spine.001'.

    Return the corrected four-bone spine chain, or None when the rig does not
    look like a Rigify-style metarig.
    """
    lookup = {name: name for name in bones}
    legacy = lookup.get('spine')
    if legacy is None or 'Hips' in found:
        return None
    chain = [legacy, lookup.get('spine.001'), lookup.get('spine.002'), lookup.get('spine.003')]
    if not all(chain):
        return None
    return dict(zip(('Hips', 'Spine', 'Spine1', 'Spine2'), chain))


def detect(armature):
    """Map canonical body bones of one armature.

    Returns (mapping, missing, conflicts): mapping is canonical name -> bone
    name in this armature, missing lists canonical names without a match, and
    conflicts lists canonical names that two bones could claim.
    """
    bones = list(armature.data.bones.keys())
    lookup = _candidates(bones)
    mapping, missing = {}, []
    for canonical in NAMES:
        bone = _match(lookup, GLB_ALIASES.get(canonical, ()), canonical)
        if bone is None:
            missing.append(canonical)
        else:
            mapping[canonical] = bone
    chain = _rigify_spine(bones, mapping)
    if chain:
        mapping.update(chain)
        missing = [name for name in NAMES if name not in mapping]
    used = {}
    conflicts = []
    for canonical, bone in mapping.items():
        if bone in used:
            conflicts.append((canonical, used[bone]))
        used[bone] = canonical
    return mapping, missing, conflicts


def rename(armature, mapping):
    """Give mapped bones their canonical names, keeping skin groups in sync.

    Only bones that are mapped, unique and not already correct are touched;
    accessory and finger bones keep their original names.
    """
    renames = {}
    taken = {bone.name for bone in armature.data.bones}
    for canonical in NAMES:
        old = mapping.get(canonical)
        if not old or old == canonical or old not in armature.data.bones:
            continue
        if canonical in taken:
            continue
        renames[old] = canonical
        taken.discard(old)
        taken.add(canonical)
    if not renames:
        return {}
    for old, new in renames.items():
        armature.data.bones[old].name = new
    groups = 0
    for child in armature.children_recursive:
        if child.type != 'MESH':
            continue
        for group in child.vertex_groups:
            if group.name in renames:
                group.name = renames[group.name]
                groups += 1
    return dict(bones=renames, vertex_groups=groups)
