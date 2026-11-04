"""Registered neutral rest frames for FBX, without changing animated geometry."""
import numpy as np
from .bvh_bind import bind_pose
from .bone_overlay import reference_frame


def vertex_normals(vertices, faces):
    """Area-weighted shared vertex normals of an outward-wound surface."""
    first = vertices[faces[:, 0]]
    cross = np.cross(vertices[faces[:, 1]] - first, vertices[faces[:, 2]] - first)
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    length = np.linalg.norm(normals, axis=1)
    if not len(normals) or (length < 1e-12).any():
        raise ValueError('The displayed body has a degenerate vertex normal')
    return normals / length[:, None]


def body_skin(rig, binding, faces, weights, basis, display, minimum=1e-6):
    """Bind the displayed body surface to the animated export skeleton.

    The viewer draws one body vertex as
    ``sum_j w_j (P_j(t) @ (v - rest_j) + p_j(t))`` with the canonical rest joint
    ``rest_j``, while FBX skinning evaluates ``NodeWorld_j(t) @ TransformLink_j^-1 @ v``
    and the exported node world is ``R_j(t) @ display_j`` for the constant export
    basis ``R_j(t) = P_j(t) @ basis_j``. Solving
    ``R_j display_j @ (basis_j display_j)^-1 = P_j`` fixes each cluster's bind
    rotation to ``basis_j @ display_j`` and its translation to the canonical rest
    joint, so a conforming reader reproduces the displayed body without any change
    to the skeleton, its bind pose or its animation channels.
    """
    from .rig_mesh import JOINT_SOURCE
    rest = np.asarray(binding['rest'], float)
    vertices = np.asarray(binding['vertices'], float)
    faces = np.asarray(faces, np.int32)
    weights = np.asarray(weights, float)
    basis = np.asarray(basis, float)
    display = np.asarray(display, float)
    count = len(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not count or not np.isfinite(vertices).all():
        raise ValueError('Invalid displayed body vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= count:
        raise ValueError('Invalid displayed body topology')
    if weights.ndim != 2 or weights.shape[0] != count or weights.shape[1] != len(rest) or not np.isfinite(weights).all():
        raise ValueError('Invalid displayed body skin weights')
    if rest.shape != (55, 3) or np.shape(basis)[1:] != (3, 3) or np.shape(basis) != np.shape(display):
        raise ValueError('Invalid export basis for the displayed body')
    if len(basis) != len(rig['rest']) or not np.allclose(rest[list(JOINT_SOURCE)], rig['rest'], rtol=0, atol=1e-5):
        raise ValueError('The displayed body and the export skeleton disagree on the rest joints')
    joint = np.zeros((count, len(basis)), np.float32)
    joint[:, :] = weights[:, list(JOINT_SOURCE)]
    # The export skeleton has no jaw or eye bones. Those joints carry the head's
    # own transform in the displayed frames, so their weight belongs to the head
    # rather than to a renormalized mix of head and neck.
    joint[:, 15] += weights[:, (22, 23, 24)].sum(1)
    joint[joint <= minimum] = 0.0
    orphan = np.flatnonzero(joint.sum(1) <= 0)
    if len(orphan):
        joint[orphan, 15] = 1.0
    joint /= joint.sum(1)[:, None]
    if not np.isfinite(joint).all():
        raise ValueError('The displayed body has an unweighted vertex')
    counts = (joint > 0).sum(1).astype(np.int32)
    offsets = np.zeros(count + 1, np.int32)
    np.cumsum(counts, out=offsets[1:])
    indices = np.zeros(int(offsets[-1]), np.int32)
    values = np.zeros(int(offsets[-1]), np.float32)
    rows, columns = np.nonzero(joint > 0)
    indices[:] = columns
    values[:] = joint[rows, columns]
    bind = np.tile(np.eye(4), (len(basis), 1, 1))
    pivot = rest[list(JOINT_SOURCE)]
    for j in range(len(basis)):
        rotation = basis[j] @ display[j]
        if np.max(abs(rotation.T @ rotation - np.eye(3))) > 1e-5 or np.linalg.det(rotation) < 0.99999:
            raise ValueError('Invalid displayed body cluster bind rotation')
        bind[j, :3, :3] = rotation
        bind[j, :3, 3] = pivot[j]
    return dict(skin_vertices=vertices.astype(np.float32), skin_faces=faces,
                skin_normals=vertex_normals(vertices, faces).astype(np.float32),
                skin_bones=indices, skin_weights=values, skin_offsets=offsets,
                skin_bind=bind)


def prepare(rig, source_rest, vertices, topology, fps):
    from .source_surface import layout, endpoints
    from .source_animation import fit
    ids, lookup = layout(topology)
    surface = endpoints(np.asarray(vertices)[None, ids], lookup)
    rest = np.asarray(source_rest)[:22]
    _, _, neutral = fit(rig, rest[None, 0], np.tile(np.eye(3), (1, 22, 1, 1)),
                        rest[None], fps=fps, surface_endpoints=surface)
    binding = bind_pose(rig)
    old = binding['rest']; parents = binding['parents']
    neutral_export = neutral[0] @ binding['basis']
    conversion = np.tile(np.eye(3), (52, 1, 1))
    limbs = [1, 2, 4, 5, 16, 17, 18, 19]
    conversion[limbs] = neutral_export[limbs].swapaxes(-1, -2)
    target = old.copy()
    display = []
    for j, p in enumerate(parents):
        if p >= 0:
            target[j] = target[p] + conversion[p].T @ (old[j] - old[p])
        children = np.flatnonzero(parents == j)
        delta = old[children].mean(0) - old[j] if len(children) else binding['ends'][j]
        if np.linalg.norm(delta) < .001:
            delta = delta + [0., .1, 0.]
        display.append(reference_frame(delta))
    display = np.asarray(display)
    target -= [target[[7,8],0].mean(), target[[10,11],1].min(), target[[7,8],2].mean()]
    neutral_frames = conversion.swapaxes(-1,-2) @ display
    # These translations must be constant in each parent's physical bone frame.
    for j, p in enumerate(parents):
        if p >= 0:
            np.testing.assert_allclose(neutral_frames[p].T @ (target[j]-target[p]),
                                       display[p].T @ (old[j]-old[p]), atol=1e-7)
    return dict(fbx_rest=target, fbx_rest_frames=neutral_frames,
                fbx_display_frames=display,
                fbx_tip_local=np.einsum('nji,nj->ni', display, binding['ends']))
