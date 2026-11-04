"""Check the live link wire format without Blender.

  runtime/python.exe -B ../tools/check-live-link.py

Starts the app's server, connects a client, publishes one pose and verifies the
handshake, the frame packet and the Y-up to Z-up conversion.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from mocap.live_link import MAGIC_JOINTS as MAGIC, NAMES, LiveLink, to_blender  # noqa: E402

PORT = 9231
report = {}


def main():
    link = LiveLink(port=PORT)
    ok, message = link.start()
    report['start'] = dict(ok=ok, message=message)
    if not ok:
        raise SystemExit(message)
    client = socket.create_connection(('127.0.0.1', PORT), timeout=3)
    client.settimeout(3)
    buffer = b''
    while b'\n' not in buffer:
        buffer += client.recv(4096)
    line, _, buffer = buffer.partition(b'\n')
    handshake = json.loads(line.decode('utf-8'))
    report['handshake'] = dict(version=handshake['version'], joints=len(handshake['names']),
                               names_match=handshake['names'] == NAMES,
                               parents=len(handshake['parents']))

    joints = np.zeros((len(NAMES), 3), dtype=float)
    joints[:, 0] = np.arange(len(NAMES)) * 0.05
    joints[:, 1] = 1.0
    joints[:, 2] = 2.0
    assert link.publish(joints, 41, 30.0, 822, 'Person 1')
    deadline = time.time() + 5
    while len(buffer) < 4 + 8 + len(NAMES) * 12 and time.time() < deadline:
        buffer += client.recv(65536)
    packet = buffer[:4 + 8 + len(NAMES) * 12]
    report['magic'] = packet[:4].decode('ascii') == MAGIC.decode('ascii')
    frame, fps = struct.unpack('<If', packet[4:12])
    values = np.array(struct.unpack('<%df' % (len(NAMES) * 3), packet[12:]), dtype=float)
    points = values.reshape(-1, 3)
    expected = to_blender(joints)
    report['packet'] = dict(frame=frame, fps=fps, joints=len(points),
                            max_error=float(np.abs(points - expected).max()),
                            first_point=points[0].round(4).tolist(),
                            expected_first=expected[0].round(4).tolist())
    report['status'] = link.status()
    link.stop()
    client.close()
    ok = (report['magic'] and report['packet']['frame'] == 41
          and report['packet']['max_error'] < 1e-6 and report['handshake']['names_match'])
    report['pass'] = bool(ok)
    print('LIVE-LINK-CHECK', json.dumps(report))
    return 0 if ok else 1


sys.exit(main())
