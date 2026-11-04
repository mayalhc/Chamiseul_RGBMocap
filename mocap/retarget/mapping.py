import numpy as np
A = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand']
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
VRM = ['hips', 'leftUpperLeg', 'rightUpperLeg', 'spine', 'leftLowerLeg', 'rightLowerLeg', 'chest', 'leftFoot', 'rightFoot', 'upperChest', 'leftToes', 'rightToes', 'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm', 'leftHand', 'rightHand']
PMX = ['下半身', '足.L', '足.R', '上半身', 'ひざ.L', 'ひざ.R', '上半身1', '足首.L', '足首.R', '上半身2', 'つま先.L', 'つま先.R', '首', '肩.L', '肩.R', '頭', '腕.L', '腕.R', 'ひじ.L', 'ひじ.R', '手首.L', '手首.R']

def body_trajectory_root(parents, mapped):
    """Use the body's existing root; independent control trees may coexist."""
    roots = set()
    for name in mapped:
        seen = set()
        while True:
            if name not in parents or name in seen:
                raise ValueError('Invalid humanoid bone hierarchy.')
            seen.add(name)
            parent = parents[name]
            if parent is None:
                roots.add(name)
                break
            name = parent
    if len(roots) != 1:
        raise ValueError('The mapped humanoid body must share one hierarchy root.')
    return next(iter(roots))

# glTF/GLB has no standard humanoid bone names. Accept the names of the common
# exporters for each canonical joint, with the namespace and separators removed.
GLB_ALIASES = {
    'Hips': ('hips', 'pelvis', 'bip01pelvis', 'hip'),
    'Spine': ('spine', 'spine01', 'spine0', 'abdomen'),
    # VRM-style rigs name the middle chest bone 'chest' and the top one
    # 'upperChest'; matching both to Spine2 made that chain ambiguous.
    'Spine1': ('spine1', 'spine02', 'spine001', 'chest'),
    'Spine2': ('spine2', 'spine03', 'spine3', 'spine002', 'upperchest'),
    'Neck': ('neck', 'neck01'),
    'Head': ('head', 'head01'),
    'LeftShoulder': ('leftshoulder', 'shoulderl', 'claviclel', 'lclavicle'),
    'RightShoulder': ('rightshoulder', 'shoulderr', 'clavicler', 'rclavicle'),
    'LeftArm': ('leftarm', 'leftupperarm', 'upperarml', 'lupperarm'),
    'RightArm': ('rightarm', 'rightupperarm', 'upperarmr', 'rupperarm'),
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


def normalized_bone(name):
    """Drop exporter namespaces (mixamorig:, Armature|) and separators."""
    return ''.join(ch for ch in name.replace('|', ':').split(':')[-1].lower() if ch.isalnum())


def glb_mapping(bones):
    """Match a glTF humanoid rig by bone name; manual mapping is not available."""
    lookup={}
    for name in bones:
        lookup.setdefault(normalized_bone(name),[]).append(name)
    resolved={}
    for canonical in NAMES:
        hits=[name for candidate in GLB_ALIASES[canonical] for name in lookup.get(candidate,[])]
        if len(hits)>1:
            raise ValueError('Ambiguous humanoid bones for '+canonical+': '+', '.join(hits))
        if hits:resolved[canonical]=hits[0]
    missing=[name for name in NAMES if name not in resolved]
    if missing:
        raise ValueError('Unsupported glTF humanoid rig. Missing bones for: '+', '.join(missing)
                         +'. Use Mixamo, VRM, Unreal or Rigify-style humanoid names.')
    if len(set(resolved.values()))!=len(NAMES):raise ValueError('Humanoid bones must have distinct mappings')
    return [resolved[name] for name in NAMES]


def body_mapping(kind,bones,semantic):
    names=set(bones)
    if kind=='glb':
        return glb_mapping(sorted(names))
    if kind=='mixamo':
        suffix={}
        for name in names:
            suffix.setdefault(name.split(':')[-1],[]).append(name)
        mapping=[suffix.get(n,[]) for n in NAMES]
        missing=[n for n,v in zip(NAMES,mapping) if len(v)!=1]
        if missing:raise ValueError('Expected a Mixamo humanoid rig. Missing or ambiguous bones: '+', '.join(missing))
        return [v[0] for v in mapping]
    mapping=[semantic.get(n) for n in VRM] if kind=='vrm' else PMX
    missing=[NAMES[i] for i,n in enumerate(mapping) if n not in names]
    if missing:raise ValueError('Unsupported humanoid mapping (manual mapping is not available yet): '+', '.join(missing))
    if len(set(mapping))!=22:raise ValueError('Humanoid bones must have distinct mappings')
    return mapping
