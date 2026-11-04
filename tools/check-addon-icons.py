"""Check that every icon the addon asks for exists in this Blender.

  blender.exe -b --factory-startup --python tools/check-addon-icons.py

Blender renames and removes icons between versions, and an unknown name raises
inside a panel's draw() rather than at registration, so this checks the source
against the running build's icon enum.
"""
import re
import sys
from pathlib import Path

import bpy

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'blender_addon/rgbmocap_bridge'
available = {item.identifier for item in
             bpy.types.UILayout.bl_rna.functions['operator'].parameters['icon'].enum_items}
pattern = re.compile(r"icon\s*=\s*['\"]([A-Za-z0-9_]+)['\"]")
missing = {}
for path in sorted(SOURCE.glob('*.py')):
    for match in pattern.finditer(path.read_text(encoding='utf-8')):
        if match.group(1) not in available:
            missing.setdefault(match.group(1), []).append(path.name)
print('CHECK-ADDON-ICONS', {'blender': bpy.app.version_string, 'icons': len(available),
                            'missing': missing}, flush=True)
if missing:
    raise SystemExit(1)
