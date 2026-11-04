"""Verify the addon's base character in an empty scene.

  blender.exe -b --python tools/verify-live-base-model.py

Starts the live link with nothing imported, as a user would, and streams the
canonical body joints. It checks the base humanoid appears, is sized from the
streamed person, and follows the joints: bone directions, hips position and the
skinned mesh all match the stream.
"""
from __future__ import annotations

import json
import math
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import bpy

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'blender_addon'))

import importlib  # noqa: E402

try:
    addon = importlib.import_module('bl_ext.user_default.rgbmocap_bridge')
except ImportError:
    addon = importlib.import_module('rgbmocap_bridge')
base_model = importlib.import_module(addon.__name__ + '.base_model')
live = importlib.import_module(addon.__name__ + '.live')

NAMES = addon.NAMES
PORT = 9237
report = {}


def base_pose(t):
    """Standing pose with swinging arms and bending knees, Z-up metres."""
    swing = math.sin(t * 6.0)
    bend = 0.35 * (0.5 + 0.5 * math.sin(t * 6.0 + 1.2))
    hips = [0.15 * math.sin(t * 3.0), 0.05 * math.sin(t * 5.0), 0.94]
    points = [None] * len(NAMES)

    def place(index, origin, direction, length):
        norm = math.sqrt(sum(value * value for value in direction)) or 1.0
        points[index] = [origin[axis] + direction[axis] / norm * length for axis in range(3)]
        return points[index]

    points[0] = hips
    spine = place(3, hips, (0.05 * swing, 0.02, 1.0), 0.12)
    spine1 = place(6, spine, (0.03 * swing, 0.0, 1.0), 0.10)
    spine2 = place(9, spine1, (0.02 * swing, 0.0, 1.0), 0.10)
    neck = place(12, spine2, (0.0, 0.0, 1.0), 0.12)
    place(15, neck, (0.0, 0.02 * swing, 1.0), 0.14)
    for side, sign in (('Left', 1.0), ('Right', -1.0)):
        shoulder_index = 13 if sign > 0 else 14
        arm_index = 16 if sign > 0 else 17
        forearm_index = 18 if sign > 0 else 19
        hand_index = 20 if sign > 0 else 21
        upleg_index = 1 if sign > 0 else 2
        leg_index = 4 if sign > 0 else 5
        foot_index = 7 if sign > 0 else 8
        toe_index = 10 if sign > 0 else 11
        shoulder = place(shoulder_index, neck, (sign, 0.0, -0.15), 0.16)
        arm = place(arm_index, shoulder, (sign, 0.0, -0.9 + 0.8 * swing * sign), 0.30)
        forearm = place(forearm_index, arm, (sign * 0.4, 0.0, -0.65 - 0.5 * abs(swing)), 0.26)
        place(hand_index, forearm, (sign * 0.2, 0.0, -1.0), 0.20)
        upleg = place(upleg_index, hips, (sign * 0.08, 0.10 * swing, -1.0), 0.42)
        leg_direction = (sign * 0.05, -math.sin(bend), -math.cos(bend))
        knee = place(leg_index, upleg, leg_direction, 0.40)
        place(foot_index, knee, (0.0, -0.8, -0.6), 0.16)
        place(toe_index, points[foot_index], (0.0, 1.0, -0.2), 0.10)
    return points


def sender(stop, go):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', PORT))
    server.listen(1)
    server.settimeout(0.5)
    client = None
    started = time.time()
    frame = 0
    try:
        while not stop.is_set():
            if client is None:
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                client.sendall((json.dumps(dict(version=2, kind='joints', names=NAMES,
                                                parents=live.PARENTS, fps=60.0, frames=600,
                                                label='Person 1'))
                                + '\n').encode('utf-8'))
            if go.is_set():
                points = base_pose(time.time() - started)
                packet = (live.MAGIC_JOINTS + struct.pack('<If', frame, 60.0)
                          + struct.pack('<%df' % (len(points) * 3),
                                        *[value for point in points for value in point]))
                try:
                    client.sendall(packet)
                except OSError:
                    client = None
                frame += 1
            time.sleep(1.0 / 60.0)
    finally:
        if client is not None:
            client.close()
        server.close()


def pump(seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        live.tick()
        time.sleep(1.0 / 60.0)


def bone_direction(armature, name):
    bone = armature.pose.bones[name]
    return (armature.matrix_world.to_3x3() @ (bone.tail - bone.head)).normalized()


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not hasattr(bpy.types.Scene, 'rgbmocap'):
        addon.register()
    settings = bpy.context.scene.rgbmocap
    settings.live_port = PORT

    stop, go = threading.Event(), threading.Event()
    threading.Thread(target=sender, args=(stop, go), daemon=True).start()
    time.sleep(0.4)

    # The user's action: Start Live Link with nothing selected or imported.
    result = bpy.ops.rgbmocap.live_start()
    report['start'] = dict(result=list(result), status=settings.live_status,
                           armatures=len([o for o in bpy.context.scene.objects if o.type == 'ARMATURE']),
                           error=live.status()['error'])
    report['created_before_frames'] = bool(base_model.find() is not None)

    go.set()
    pump(1.2)
    # Stop the stream, then settle so the compared packet is the applied one.
    stop.set()
    time.sleep(0.3)
    pump(0.2)
    armature = base_model.find()
    meshes = [o for o in bpy.context.scene.objects
              if o.type == 'MESH' and o.get(base_model.MARK)]
    bones = sorted(bone.name for bone in armature.data.bones) if armature else []
    with live._CLIENT.lock:
        points = list(live._CLIENT.latest) if live._CLIENT.latest else None

    errors = []
    for index, name in enumerate(NAMES):
        child = live.CHAIN.get(name)
        if child is None:
            continue
        expected = [points[NAMES.index(child)][axis] - points[index][axis] for axis in range(3)]
        length = math.sqrt(sum(value * value for value in expected)) or 1.0
        dot = max(-1.0, min(1.0, sum(bone_direction(armature, name)[axis] * expected[axis] / length
                                     for axis in range(3))))
        errors.append((name, math.degrees(math.acos(dot))))
    worst = max(errors, key=lambda item: item[1])
    hips = armature.matrix_world @ armature.pose.bones['Hips'].head
    hips_error = math.sqrt(sum((hips[axis] - points[0][axis]) ** 2 for axis in range(3)))

    # Bone lengths must be the streamed person's, not the default figure's.
    streamed_hips = math.dist(points[0], points[3])
    rest_hips = armature.data.bones['Hips'].length
    thigh = armature.data.bones['LeftUpLeg'].length
    streamed_thigh = math.dist(points[1], points[4])

    depsgraph = bpy.context.evaluated_depsgraph_get()
    body = meshes[0].evaluated_get(depsgraph)
    data = body.to_mesh()
    try:
        heights = [vertex.co.z for vertex in data.vertices]
        spread = (max(heights) - min(heights)) if heights else 0.0
        posed = [tuple(round(value, 3) for value in vertex.co) for vertex in data.vertices[:40]]
    finally:
        body.to_mesh_clear()

    report['base'] = dict(armature=armature.name if armature else '',
                          meshes=len(meshes), bones=len(bones), canonical=sum(
                              1 for name in NAMES if name in bones),
                          vertices=len(meshes[0].data.vertices) if meshes else 0)
    report['stream'] = dict(received=live.status()['received'], applied=live.status()['applied'],
                            frame=live.status()['frame'], error=live.status()['error'],
                            target=live.status()['target'],
                            base_default=bool(armature.get(base_model.DEFAULT_MARK)) if armature else None)
    report['directions'] = dict(bones=len(errors), worst_deg=round(worst[1], 4),
                                worst_bone=worst[0])
    report['hips'] = dict(error_m=round(hips_error, 6))
    report['proportions'] = dict(hips_bone=round(rest_hips, 4), streamed_hips=round(streamed_hips, 4),
                                 thigh=round(thigh, 4), streamed_thigh=round(streamed_thigh, 4))
    report['mesh'] = dict(height=round(spread, 3), sample=posed[:2])
    live.stop()
    # The panel button works on its own, with no stream to size it.
    bpy.ops.object.select_all(action='DESELECT')
    button = bpy.ops.rgbmocap.load_base()
    report['button'] = dict(result=list(button), armatures=len(
        [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']),
        bones=len(base_model.find().data.bones) if base_model.find() else 0)

    ok = (report['created_before_frames'] and armature is not None and len(meshes) == 1
          and report['button']['result'] == ['FINISHED'] and report['button']['bones'] == len(NAMES)
          and report['base']['canonical'] == len(NAMES)
          and report['stream']['applied'] > 5 and worst[1] < 0.5 and hips_error < 0.005
          and abs(rest_hips - streamed_hips) < 0.005 and abs(thigh - streamed_thigh) < 0.005
          and 0.5 < spread < 2.5)
    report['pass'] = bool(ok)
    target = BASE / 'screenshots' / 'live-base-model-report.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VERIFY-LIVE-BASE', json.dumps(report), flush=True)
    if not ok:
        raise SystemExit(1)


main()
