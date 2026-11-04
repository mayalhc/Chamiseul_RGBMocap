"""Verify the Blender side of the live link against a fake app sender.

  blender.exe -b --python tools/verify-live-link.py

Imports a real Mixamo character, streams a moving synthetic pose over the same
wire format the app uses, applies every frame to the rig, then measures how far
each mapped bone direction and the hips position are from the streamed joints.
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
live = importlib.import_module(addon.__name__ + '.live')
mapping = importlib.import_module(addon.__name__ + '.mapping')

NAMES = addon.NAMES
PARENTS = live.PARENTS
CHARACTER = Path('C:/Users/mayal/Downloads/Ch14_nonPBR.fbx')
PORT = 9232
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


def app_sender(stop):
    """Mimic the app: handshake line, then RGMF packets at ~60 Hz."""
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
                client.sendall((json.dumps(dict(version=1, names=NAMES, parents=PARENTS,
                                                fps=60.0, frames=600, label='Person 1'))
                                + '\n').encode('utf-8'))
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


def armature():
    return next(obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')


def bone_direction(armature_object, name):
    bone = armature_object.pose.bones[name]
    return (armature_object.matrix_world.to_3x3() @ (bone.tail - bone.head)).normalized()


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not hasattr(bpy.types.Scene, 'rgbmocap'):
        addon.register()
    bpy.ops.import_scene.fbx(filepath=str(CHARACTER), automatic_bone_orientation=False,
                             use_prepost_rot=True)
    armature_object = armature()
    bpy.ops.object.select_all(action='DESELECT')
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    found, missing, _ = addon.describe(armature_object)
    report['mapping'] = dict(matched=len(found), missing=missing)

    stop = threading.Event()
    sender = threading.Thread(target=app_sender, args=(stop,), daemon=True)
    sender.start()
    time.sleep(0.4)
    if not live.start('127.0.0.1', PORT):
        raise SystemExit('no connection: ' + live.status()['error'])
    report['handshake'] = live.status()['handshake']

    # The background timer does not run reliably in `-b`, so step it directly.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        live.tick()
        time.sleep(1.0 / 60.0)

    # Record live poses into an action, keeping the streamed pose per key.
    captured = {}
    bpy.context.scene.frame_set(1)
    live.record_start(armature_object, 1)
    deadline = time.time() + 1.5
    last_key = 0
    while time.time() < deadline:
        with live._CLIENT.lock:
            before = live._CLIENT.latest
        live.tick()
        info = live.recording()
        if info['last'] != last_key:
            with live._CLIENT.lock:
                after = live._CLIENT.latest
            # Only keep pairs where no newer packet arrived during the tick, so
            # the comparison is against the exact pose that was keyed.
            if after is before:
                captured[info['last']] = after
            last_key = info['last']
        time.sleep(1.0 / 60.0)
    recorded = live.record_stop()
    status = live.status()
    report['status'] = dict(received=status['received'], applied=status['applied'],
                            frame=status['frame'], fps=status['fps'])
    report['error'] = status['error']
    # Stop the stream, then apply the final packet so the pose and the packet
    # being compared describe the same frame rather than the stream lag.
    stop.set()
    time.sleep(0.3)
    for _ in range(8):
        live.tick()
        time.sleep(1.0 / 60.0)
    with live._CLIENT.lock:
        points = live._CLIENT.latest
    errors = []
    for index, name in enumerate(NAMES):
        child = live.CHAIN.get(name)
        if child is None or name not in found:
            continue
        expected = [points[NAMES.index(child)][axis] - points[index][axis] for axis in range(3)]
        length = math.sqrt(sum(value * value for value in expected)) or 1.0
        expected = [value / length for value in expected]
        actual = bone_direction(armature_object, found[name])
        dot = max(-1.0, min(1.0, sum(actual[axis] * expected[axis] for axis in range(3))))
        errors.append((name, math.degrees(math.acos(dot))))
    worst = max(errors, key=lambda item: item[1])
    hips = armature_object.matrix_world @ armature_object.pose.bones[found['Hips']].head
    hips_error = math.sqrt(sum((hips[axis] - points[0][axis]) ** 2 for axis in range(3)))
    report['directions'] = dict(bones=len(errors), worst_deg=round(worst[1], 4),
                                worst_bone=worst[0], mean_deg=round(
                                    sum(value for _, value in errors) / len(errors), 4))
    report['hips'] = dict(error_m=round(hips_error, 6))

    action = armature_object.animation_data.action if armature_object.animation_data else None
    curves = list(live._iter_fcurves(action)) if action else []
    keys = sum(len(curve.keyframe_points) for curve in curves)
    interpolation = sorted({point.interpolation for curve in curves
                            for point in curve.keyframe_points})
    report['record'] = dict(action=action.name if action else '', requested=recorded['action'],
                            key_sets=recorded['keys'], bones=recorded['bones'],
                            curves=len(curves), keys=keys, interpolation=interpolation,
                            start=recorded['start'], last=recorded['last'],
                            scene_range=[bpy.context.scene.frame_start, bpy.context.scene.frame_end],
                            captured=len(captured))

    playback = []
    sample = sorted(captured)[::max(1, len(captured) // 5)] if captured else []
    for scene_frame in sample:
        bpy.context.scene.frame_set(scene_frame)
        bpy.context.view_layer.update()
        points_at = captured[scene_frame]
        worst_playback = 0.0
        for index, name in enumerate(NAMES):
            child = live.CHAIN.get(name)
            if child is None or name not in found:
                continue
            expected = [points_at[NAMES.index(child)][axis] - points_at[index][axis] for axis in range(3)]
            length = math.sqrt(sum(value * value for value in expected)) or 1.0
            expected = [value / length for value in expected]
            actual = bone_direction(armature_object, found[name])
            dot = max(-1.0, min(1.0, sum(actual[axis] * expected[axis] for axis in range(3))))
            worst_playback = max(worst_playback, math.degrees(math.acos(dot)))
        playback.append((scene_frame, worst_playback))
    report['playback'] = dict(frames=len(playback),
                              worst_deg=round(max((item[1] for item in playback), default=0.0), 4),
                              sample=[[frame, round(value, 4)] for frame, value in playback[:6]])

    mesh = next(obj for obj in bpy.context.scene.objects if obj.type == 'MESH')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    data = evaluated.to_mesh()
    try:
        posed = [vertex.co.copy() for vertex in data.vertices[:200]]
    finally:
        evaluated.to_mesh_clear()
    report['mesh_deformed'] = any(vertex.z != 0 or vertex.x != 0 for vertex in posed)

    live.stop()
    ok = (worst[1] < 0.05 and hips_error < 0.005 and status['applied'] > 5
          and report['mapping']['matched'] == len(NAMES)
          and report['record']['key_sets'] > 20 and report['record']['keys'] > 400
          and report['record']['interpolation'] == ['LINEAR']
          and report['playback']['worst_deg'] < 0.05)
    report['pass'] = bool(ok)
    target = Path(__file__).resolve().parents[1] / 'screenshots' / 'live-link-report.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('VERIFY-LIVE', json.dumps(report))
    if not ok:
        raise SystemExit(1)


main()
