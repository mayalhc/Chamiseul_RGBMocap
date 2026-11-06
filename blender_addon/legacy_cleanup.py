# Copyright (c) 2026 Chamiseul. All rights reserved.
# Proprietary: redistribution prohibited. See LICENSE-CHAMISEUL.md in the repository root.
"""Remove a legacy scripts/addons copy of the bridge, so the extension is the only one.

Run with Blender: blender -b --python blender_addon/legacy_cleanup.py
"""
import shutil
from pathlib import Path

import addon_utils
import bpy

try:
    addon_utils.disable('rgbmocap_bridge')
except Exception as error:  # not installed as a legacy add-on
    print('LEGACY-CLEANUP no legacy module:', error)

target = Path(bpy.utils.user_resource('SCRIPTS', path='addons')) / 'rgbmocap_bridge'
if target.exists():
    shutil.rmtree(target)
    print('LEGACY-CLEANUP removed:', target)
else:
    print('LEGACY-CLEANUP nothing at:', target)
bpy.ops.wm.save_userpref()
