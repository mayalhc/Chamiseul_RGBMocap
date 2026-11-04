"""Live body solver: SAM 3D Body keypoints to the app's own rig, per frame.

The webcam worker runs SAM 3D Body on each camera frame; its MHR70 keypoints
are camera-space metres. This module maps them onto the app's 22 body joints
(the same ordering every preview and export uses) and aims the base
character's own rig at them: the solved joints keep their measured directions
and take the rig's bone lengths, and each bone inherits its roll down the
chain. Those world frames are what the 3D view skins the base character mesh
with and what the bone overlay draws - no separate stand-in skeleton.

Pure numpy; importable by both the GUI and the live worker process.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .mixamo_rig import BODY_PARENTS, align, fit_frame, fk, skeleton, unit
from .source_animation import source_rig

# MHR70: SAM 3D Body's 70 whole-body keypoints (third_party fast-sam-3d-body
# sam_3d_body/metadata/mhr70.py). Only the body points drive the rig.
_MHR_LEFT_SHOULDER, _MHR_RIGHT_SHOULDER = 5, 6
_MHR_LEFT_ELBOW, _MHR_RIGHT_ELBOW = 7, 8
_MHR_LEFT_HIP, _MHR_RIGHT_HIP = 9, 10
_MHR_LEFT_KNEE, _MHR_RIGHT_KNEE = 11, 12
_MHR_LEFT_ANKLE, _MHR_RIGHT_ANKLE = 13, 14
_MHR_LEFT_TOE, _MHR_RIGHT_TOE = 15, 18
_MHR_RIGHT_WRIST, _MHR_LEFT_WRIST = 41, 62
_MHR_NOSE, _MHR_NECK = 0, 69


def base_rest(smplx_path):
    """The 22 template rest joints (metres, Y-up, pelvis centred) of SMPL-X."""
    with np.load(smplx_path, allow_pickle=False) as data:
        joints = (data['J_regressor'][:22] @ data['v_template']).astype(np.float64)
    joints -= joints[0]
    return joints


def build_live_rig(rest):
    """The app rig for live transfer: same construction every preview uses."""
    return source_rig(skeleton(rest), rest)


def cached_rest(root, smplx_path):
    """The live rest, cached as a tiny npz beside the app cache."""
    cache = Path(root) / 'cache' / 'live-skeleton.npz'
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as data:
                if data['rest'].shape == (22, 3):
                    return data['rest']
        except (OSError, ValueError, KeyError):
            pass
    rest = base_rest(smplx_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + '.partial.npz')
    np.savez(temporary, rest=rest)
    temporary.replace(cache)
    return rest


class LiveBody:
    """Everything fixed across a live session, plus the running pole state.

    ``rest`` is the 22-joint SMPL-X template rest (pelvis centred, Y-up);
    ``rig`` is the app's 52-bone fixed-length rig built from it.
    """

    def __init__(self, rest=None, rig=None):
        if rig is not None:
            # The take's OWN character: its prepared rig carries that person's
            # proportions, so the live body is the same character the result
            # view presents instead of a neutral stand-in.
            self.rig = {key: (value.copy() if isinstance(value, np.ndarray) else value)
                        for key, value in rig.items()}
            self.rest = np.asarray(self.rig['rest'][:22], dtype=np.float64)
        else:
            self.rest = np.asarray(rest, dtype=np.float64)
            self.rig = build_live_rig(self.rest)
        self.pole_state = {}
        # Interpolation fractions of the unmeasured trunk/clavicle joints,
        # read off the rest skeleton once and applied to the measured chains.
        neck = self.rest[12]
        self.spine_fraction = {}
        span = neck - self.rest[0]
        length = float(span @ span)
        for j in (3, 6, 9):
            fraction = 0.3 + 0.25 * {3: 0, 6: 1, 9: 2}[j]
            if length > 1e-9:
                fraction = float(np.clip((self.rest[j] - self.rest[0]) @ span / length, 0.05, 0.95))
            self.spine_fraction[j] = fraction
        self.shoulder_fraction = {}
        for shoulder, arm in ((13, 16), (14, 17)):
            span = self.rest[arm] - self.rest[12]
            length = float(span @ span)
            fraction = 0.45
            if length > 1e-9:
                fraction = float(np.clip((self.rest[shoulder] - self.rest[12]) @ span / length, 0.1, 0.9))
            self.shoulder_fraction[shoulder] = fraction
        self.head_length = float(np.linalg.norm(self.rest[15] - self.rest[12]))
        # Per-joint camera depth of the previous solve: the branch memory that
        # keeps the two-bone ray roots from flipping limbs toward the camera.
        self.joint_depths = np.zeros(22)
        # The rest offsets the chain solver rotates, root-relative like the rig.
        self.chain_rest = self.rig['rest'][:22].copy()
        # The torso segment the planar depth anchor rides on.
        self.torso_rest = float(np.linalg.norm(
            0.5 * (self.chain_rest[13] + self.chain_rest[14]) - self.chain_rest[0]))

    def positions_from_2d(self, keypoints, focal, principal, root_depth=None,
                          depth_hints=None, hint_scale=1.0):
        """COCO-23 2D keypoints -> the app's 22 Y-up world joints.

        The root depth is searched so the fixed-length leg chains best reach
        their measured 2D rays; every measured joint then lands on its ray at
        the depth that keeps its rest bone length from the solved parent. A
        ray the bone cannot reach pulls the joint as far along it as the bone
        allows - lengths stay fixed (the rig re-derives them exactly), only
        the direction is trusted.
        """
        keypoints = np.asarray(keypoints, dtype=np.float64)
        confident = keypoints[:, 2] > 0.3
        rest = self.rest
        chain = self.chain_rest
        hips_px = 0.5 * (keypoints[11, :2] + keypoints[12, :2])
        clavicle_rest = 0.5 * (chain[13] + chain[14])
        torso_rest = float(np.linalg.norm(clavicle_rest - chain[0]))
        shoulders_px = 0.5 * (keypoints[5, :2] + keypoints[6, :2])
        pixels = max(8.0, float(np.linalg.norm(shoulders_px - hips_px)))
        anchor = focal * torso_rest / pixels

        def ray(index):
            u = (keypoints[index, 0] - principal[0]) / focal
            v = (keypoints[index, 1] - principal[1]) / focal
            direction = np.array([u, v, 1.0])
            return direction / np.linalg.norm(direction)

        def reach(parent, direction, length, joint):
            """(point, shortfall): the child on its ray at the fixed length,
            pulled along the ray as far as the bone allows when out of reach.

            The two sphere roots sit symmetrically about the ray's closest
            approach, and always taking the near one folds the limb toward
            the camera. The branch follows the joint's own previous depth;
            with no memory yet, the root nearer the torso depth wins.
            """
            dot = float(direction @ parent)
            disc = dot * dot - float(parent @ parent) + length * length
            if disc <= 0.0:
                target = direction * dot
                delta = target - parent
                distance = float(np.linalg.norm(delta))
                if distance < 1e-9:
                    return target, length
                return parent + delta * (length / distance), length - distance
            near, far = dot - math.sqrt(disc), dot + math.sqrt(disc)
            if near <= 0.0:
                return direction * far, 0.0
            # The two roots sit symmetrically on the ray; the limb's observed
            # 2D direction disambiguates: a ray descending in the image (a leg
            # or a hanging arm) continues DOWN and AWAY, so the farther root
            # is the joint; an ascending ray (a raised hand) takes the nearer.
            # A real-3D depth hint (SAM 3D Body calibration) decides the
            # branch outright; otherwise the ray's image direction does: a
            # descending ray (a leg, a hanging arm) continues DOWN and AWAY,
            # so the farther root is the joint; an ascending ray (a raised
            # hand) takes the nearer one.
            hint = hint_scale * float(depth_hints[joint]) if depth_hints is not None else 0.0
            if hint > 0.0:
                pick = near if abs(near - hint) <= abs(far - hint) else far
            elif direction[1] > 0.03:
                pick = far
            elif direction[1] < -0.03:
                pick = near
            else:
                pick = far
            return direction * pick, 0.0

        ray_hips = 0.5 * (ray(11) + ray(12))
        ray_hips /= np.linalg.norm(ray_hips)
        state = {'score': 0.0}

        updates = {}

        def solve_ray(parent, direction, length, joint):
            point, miss = reach(parent, direction, length, joint)
            state['score'] += miss * miss
            updates[joint] = float(point[2])
            return point

        LEGS = ((11, 1, 4, 7, 10), (12, 2, 5, 8, 11))

        def solve_body(depth_root):
            camera = np.empty((22, 3))
            camera[0] = ray_hips * depth_root
            for hip_kp, upleg, leg, foot, toe in LEGS:
                camera[upleg] = solve_ray(camera[0], ray(hip_kp), float(np.linalg.norm(chain[upleg] - chain[0])), upleg)
                camera[leg] = solve_ray(camera[upleg], ray(13 if hip_kp == 11 else 14),
                                        float(np.linalg.norm(chain[leg] - chain[upleg])), leg)
                camera[foot] = solve_ray(camera[leg], ray(15 if hip_kp == 11 else 16),
                                         float(np.linalg.norm(chain[foot] - chain[leg])), foot)
                camera[toe] = solve_ray(camera[foot], ray(17 if hip_kp == 11 else 20),
                                        float(np.linalg.norm(chain[toe] - chain[foot])), toe)
            return camera

        # The root depth. Monocular keypoints cannot pin absolute depth, so a
        # calibrated scale (from SAM 3D Body's real 3D, refreshed occasionally)
        # turns the planar anchor into the root depth; before the first
        # calibration the width-of-the-body residual stands in. The search
        # spans far: a crouched first frame shrinks the planar anchor a lot.
        # Depth tracking: the person's distance changes slowly, so once a
        # depth exists the search only refines around it; the width-of-the-
        # body residual picks the candidate (bone lengths are fixed by
        # construction, so at the right depth the left/right separations
        # match the rest skeleton).
        pairs = ((1, 2), (4, 5), (7, 8))

        def score_depth(candidate):
            state['score'] = 0.0
            solved = solve_body(candidate)
            total = 0.0
            for left, right in pairs:
                error = float(np.linalg.norm(solved[left] - solved[right])
                              - np.linalg.norm(chain[left] - chain[right]))
                total += error * error
            return total

        if root_depth is not None and root_depth > 0.1:
            best_depth, best_score = float(root_depth), None
            for factor in np.linspace(0.6, 1.6, 21):
                candidate = float(factor * root_depth)
                value = score_depth(candidate)
                if best_score is None or value < best_score:
                    best_score, best_depth = value, candidate
        else:
            best_depth, best_score = anchor, None
            for factor in np.linspace(0.35, 4.5, 42):
                candidate = float(factor * anchor)
                value = score_depth(candidate)
                if best_score is None or value < best_score:
                    best_score, best_depth = value, candidate
        camera = solve_body(best_depth)
        # Upper body hangs off the solved torso.
        mid = 0.5 * (keypoints[5, :2] + keypoints[6, :2])
        mid_ray = np.array([(mid[0] - principal[0]) / focal,
                            (mid[1] - principal[1]) / focal, 1.0])
        mid_ray /= np.linalg.norm(mid_ray)
        shoulders_mid = solve_ray(camera[0], mid_ray, torso_rest, 9)
        for kp, arm in ((5, 16), (6, 17)):
            camera[arm] = solve_ray(shoulders_mid, ray(kp), float(np.linalg.norm(
                chain[arm] - chain[9])), arm)
        for kp, elbow, wrist, upper in ((7, 18, 20, 16), (8, 19, 21, 17)):
            camera[elbow] = solve_ray(camera[upper], ray(kp), float(np.linalg.norm(chain[elbow] - chain[upper])), elbow)
            camera[wrist] = solve_ray(camera[elbow], ray(9 if kp == 7 else 10),
                                      float(np.linalg.norm(chain[wrist] - chain[elbow])), wrist)
        neck_axis = unit(rest[12] - clavicle_rest)
        camera[12] = shoulders_mid + neck_axis * float(np.linalg.norm(rest[12] - clavicle_rest))
        nose = solve_ray(camera[12], ray(0), float(np.linalg.norm(rest[15] - rest[12])), 15)
        head_axis = nose - camera[12]
        head_length = float(np.linalg.norm(head_axis))
        camera[15] = camera[12] + (head_axis / head_length if head_length > 1e-6 else neck_axis) * self.head_length
        for j, fraction in self.spine_fraction.items():
            camera[j] = camera[0] + fraction * (camera[12] - camera[0])
        for shoulder, arm in ((13, 16), (14, 17)):
            camera[shoulder] = camera[12] + self.shoulder_fraction[shoulder] * (camera[arm] - camera[12])
        # The width residual of this frame - the solver's own quality score.
        quality = 0.0
        for left, right in ((1, 2), (4, 5), (7, 8)):
            error = float(np.linalg.norm(camera[left] - camera[right])
                          - np.linalg.norm(chain[left] - chain[right]))
            quality += error * error
        # Commit the branch memory only on a sane solve; a collapsed frame
        # must not poison the next one.
        if quality < 0.03:
            for joint, depth in updates.items():
                self.joint_depths[joint] = depth
        # OpenCV (x, y down, z forward) -> the app's world. The app's takes
        # face -Z while the camera sees the person's front, so the world turns
        # the camera frame around: a proper 180-degree turn about the view axis.
        return camera * np.array([-1.0, -1.0, 1.0]), quality

    def positions(self, keypoints, cam_t):
        """MHR70 camera-space keypoints -> the app's 22 Y-up world joints.

        The camera space is OpenCV (x right, y down, z forward); the app is
        Y-up metres with the feet resting on the floor.
        """
        keypoints = np.asarray(keypoints, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64)
        points = np.empty((22, 3))
        points[0] = 0.5 * (keypoints[_MHR_LEFT_HIP] + keypoints[_MHR_RIGHT_HIP])
        # The leg roots ride the MEASURED thighs: the SAM hip keypoints sit at
        # pelvis height, so a pelvis-to-hip-keypoint offset renders as a thigh
        # pointing sideways or up. Point the root down its own thigh instead.
        for hip_kp, upleg, knee_kp in ((_MHR_LEFT_HIP, 1, _MHR_LEFT_KNEE),
                                       (_MHR_RIGHT_HIP, 2, _MHR_RIGHT_KNEE)):
            thigh = keypoints[knee_kp] - keypoints[hip_kp]
            length = float(np.linalg.norm(thigh))
            points[upleg] = points[0] + (thigh / length if length > 1e-6
                                         else np.array([0.0, -1.0, 0.0])) * float(
                np.linalg.norm(self.chain_rest[upleg] - self.chain_rest[0]))
        neck = keypoints[_MHR_NECK]
        for j, fraction in self.spine_fraction.items():
            points[j] = points[0] + fraction * (neck - points[0])
        points[4] = keypoints[_MHR_LEFT_KNEE]
        points[5] = keypoints[_MHR_RIGHT_KNEE]
        points[7] = keypoints[_MHR_LEFT_ANKLE]
        points[8] = keypoints[_MHR_RIGHT_ANKLE]
        points[10] = keypoints[_MHR_LEFT_TOE]
        points[11] = keypoints[_MHR_RIGHT_TOE]
        points[12] = neck
        head_axis = keypoints[_MHR_NOSE] - neck
        length = float(np.linalg.norm(head_axis))
        points[15] = neck + (head_axis / length if length > 1e-6
                             else np.array([0.0, 1.0, 0.0])) * self.head_length
        for side, (shoulder, arm, shoulder_kp, elbow_kp) in enumerate(
                ((13, 16, _MHR_LEFT_SHOULDER, _MHR_LEFT_ELBOW),
                 (14, 17, _MHR_RIGHT_SHOULDER, _MHR_RIGHT_ELBOW))):
            upper = keypoints[shoulder_kp]
            points[shoulder] = points[12] + self.shoulder_fraction[shoulder] * (upper - points[12])
            # The upper arm points down its own measured arm, like the legs.
            arm_dir = keypoints[elbow_kp] - upper
            length = float(np.linalg.norm(arm_dir))
            points[arm] = points[shoulder] + (arm_dir / length if length > 1e-6
                                              else np.array([0.0, -1.0, 0.0])) * float(
                np.linalg.norm(self.chain_rest[arm] - self.chain_rest[shoulder]))
        points[18] = keypoints[_MHR_LEFT_ELBOW]
        points[19] = keypoints[_MHR_RIGHT_ELBOW]
        points[20] = keypoints[_MHR_LEFT_WRIST]
        points[21] = keypoints[_MHR_RIGHT_WRIST]
        # OpenCV (x, y down, z forward) -> the app's world. The app's takes
        # face -Z while the camera sees the person's front, so the world turns
        # the camera frame around: a proper 180-degree turn about the view axis.
        return points * np.array([-1.0, -1.0, 1.0])

    def aim(self, positions):
        """The base character's world frames for one solved pose.

        The rig is the base character's own skeleton, so its bone lengths are
        fixed: every bone keeps its measured direction and takes the rig's own
        length, walking the skeleton parents first. The hips take the best
        rigid rotation of the pelvis (its two thighs, the spine, and the
        shoulder line, which is what fixes the facing); every other joint then
        takes the minimal rotation that puts its own bone on the measured
        direction, so roll is inherited down the chain rather than invented.
        """
        rig = self.rig
        rest = rig['rest']
        parents = rig['parents']
        points = np.asarray(positions, dtype=np.float64).copy()
        for j, parent in enumerate(BODY_PARENTS):
            if parent < 0:
                continue
            direction = points[j] - points[parent]
            length = float(np.linalg.norm(direction))
            bone = float(np.linalg.norm(self.chain_rest[j] - self.chain_rest[parent]))
            points[j] = points[parent] + (direction * (bone / length) if length > 1e-6
                                          else self.chain_rest[j] - self.chain_rest[parent])
        world = np.tile(np.eye(3), (52, 1, 1))
        # The pelvis yaw is unconstrained by the thighs and spine alone (the
        # rig points both thighs straight down at rest), so the base
        # character's own lateral axis carries it: the collar line keeps the
        # body facing the person even while the arms move.
        lateral = points[13] - points[14]
        if np.linalg.norm(lateral) < 1e-3:
            lateral = points[16] - points[17]
        reference = np.stack([unit(self.chain_rest[j] - self.chain_rest[0]) for j in (1, 2, 3)]
                             + [np.array([1.0, 0.0, 0.0])])
        desired = np.stack([unit(points[j] - points[0]) for j in (1, 2, 3)]
                           + [unit(lateral)])
        world[0] = fit_frame(reference, desired[None])[0]
        # A joint's frame drives its OUTGOING bones, so each frame is aimed at
        # the joints below it: one child takes the minimal rotation (roll stays
        # inherited from the parent), a branching joint best-fits all of them.
        below = {}
        for j, parent in enumerate(BODY_PARENTS):
            if parent >= 0:
                below.setdefault(int(parent), []).append(j)
        for j, parent in enumerate(parents):
            if parent < 0 or j == 0:
                continue
            if j >= 22:
                # Hands come from the body encoder alone; the base character
                # keeps its own relaxed hand instead of a splayed one.
                world[j] = world[parent]
                continue
            legs = below.get(j)
            if not legs:
                # A leaf (a toe, the head, a hand) keeps its parent's frame.
                world[j] = world[parent]
                continue
            if len(legs) == 1:
                child = legs[0]
                shown = world[parent] @ (rest[child] - rest[j])
                world[j] = align(shown, points[child] - points[j]) @ world[parent]
            else:
                shown = np.stack([world[parent] @ (rest[child] - rest[j]) for child in legs])
                wanted = np.stack([points[child] - points[j] for child in legs])
                world[j] = fit_frame(shown, wanted[None])[0] @ world[parent]
        local = world.copy()
        for j, parent in enumerate(parents):
            if parent >= 0:
                local[j] = world[parent].swapaxes(-1, -2) @ world[j]
        points, world = fk(rig, points[0:1], local[None])
        return local, points[0], world[0]

    def rig_row(self, positions):
        """One (52, 4, 4) base-character frame, plus the joints it places."""
        local, points, world = self.aim(positions)
        row = np.tile(np.eye(4), (52, 1, 1))
        row[:, :3, :3] = world
        row[:, :3, 3] = points
        return row, points[:22]


class Smoother:
    """Per-joint exponential smoothing; the floor offset eases separately."""

    def __init__(self, alpha=0.55, floor_alpha=0.2):
        self.alpha = alpha
        self.floor_alpha = floor_alpha
        self.state = None
        self.floor = None

    def step(self, points):
        if self.state is None:
            self.state = points.copy()
        else:
            self.state += self.alpha * (points - self.state)
        floor = float(min(points[7, 1], points[8, 1], points[10, 1], points[11, 1]))
        if self.floor is None:
            self.floor = floor
        else:
            self.floor += self.floor_alpha * (floor - self.floor)
        return self.state - np.array([0.0, self.floor, 0.0])
