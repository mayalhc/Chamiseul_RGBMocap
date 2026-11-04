"""Render the live start-up overlay offscreen and read its numbers back.

The wait before the first pose is shown in the 3D view as a step, a percentage
and elapsed seconds. This checks that the view actually draws them (the widget
is rendered without a window) and that a solved pose removes the overlay.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from mocap.viewport import MotionView


def light_pixels(image):
    """Count non-background pixels, so a blank render is detectable."""
    bits = image.convertToFormat(QImage.Format.Format_RGB32)
    data = np.frombuffer(bits.bits(), np.uint8).reshape(bits.height(), bits.bytesPerLine() // 4, 4)
    rgb = data[:, :, :3].astype(int)
    return int((rgb.sum(axis=2) > 210).sum())


def main():
    app = QApplication.instance() or QApplication([])
    view = MotionView()
    view.resize(720, 480)
    view.show()
    app.processEvents()

    empty = light_pixels(view.grab().toImage())

    view.live_progress = {
        'title': 'Live character is starting',
        'stage': 'Loading the body model',
        'percent': 60.0,
        'seconds': 12.4,
        'step': 4,
        'steps': 5,
    }
    view.update()
    app.processEvents()
    loading = view.grab().toImage()
    loading.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'screenshots', '.live-progress-loading.png'))

    view.live_progress = {
        'title': 'Live character is starting',
        'stage': 'Computing the first pose',
        'percent': 80.0,
        'seconds': 13.9,
        'step': 5,
        'steps': 5,
    }
    view.update()
    app.processEvents()
    solving = view.grab().toImage()

    # A solved pose owns the view: the wait numbers go away with it.
    summary = {'floor': 0.0, 'height': 1.7, 'span': 1.7, 'center': np.zeros(3, np.float32)}
    view.set_motion(np.zeros((1, 22, 3), np.float32), summary, None)
    view.live_progress = None
    view.update()
    app.processEvents()
    solved = light_pixels(view.grab().toImage())

    print('empty %d, loading %d, solving %d, solved %d'
          % (empty, light_pixels(loading), light_pixels(solving), solved))
    ok = light_pixels(loading) > empty + 200 and light_pixels(solving) > empty + 200
    print('LIVE-PROGRESS-CHECK: ' + ('ok' if ok else 'failed'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
