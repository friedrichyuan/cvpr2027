"""Self-contained, pure-NumPy MANO forward kinematics.

This module re-renders the MANO hand model from the pose/shape parameters stored
in the delivery's ``hands.npz`` without any heavy external dependency (no
deep-learning framework required). It only needs ``numpy`` and the vendored model
files under ``assets/mano/MANO_RIGHT.npz`` / ``MANO_LEFT.npz``.

The model files were converted once from the standard MANO model files into a
dependency-free NumPy ``.npz`` format so they load anywhere. See ``README.md``
for provenance.

The forward pass is a standard MANO Linear Blend Skinning with shape and pose
blend shapes. In addition to the 778 mesh vertices it returns 21 keypoints
(16 MANO skeleton joints + 5 fingertips sampled from the mesh) for lightweight
2D/3D skeleton drawing.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

# MANO mesh vertex indices of the five fingertips (standard MANO tip vertices).
# Order matters: the extra tip joints are appended in this order before being
# remapped to OpenPose order.
FINGERTIP_VERTEX_IDS: Dict[str, int] = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}

# MANO->OpenPose joint remapping. Applied to the 21 = [16 MANO joints,
# 5 fingertips] stacked array to produce the standard 21-keypoint OpenPose hand
# order used by the reference keypoint convention.
MANO_TO_OPENPOSE: Tuple[int, ...] = (
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20,
)

# OpenPose 21-keypoint hand skeleton (standard hand keypoint connectivity).
HAND_SKELETON: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # little
)

_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "assets", "mano")


def _rodrigues(axisang: np.ndarray) -> np.ndarray:
    """Batched axis-angle -> rotation matrix.

    Args:
        axisang: ``(N, 3)`` axis-angle vectors.

    Returns:
        ``(N, 3, 3)`` rotation matrices.
    """
    axisang = np.asarray(axisang, dtype=np.float64)
    angle = np.linalg.norm(axisang + 1e-8, axis=1, keepdims=True)
    axis = axisang / angle
    cos = np.cos(angle)[:, :, None]   # (N,1,1)
    sin = np.sin(angle)[:, :, None]
    n = axisang.shape[0]
    rx, ry, rz = axis[:, 0], axis[:, 1], axis[:, 2]
    K = np.zeros((n, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -rz
    K[:, 0, 2] = ry
    K[:, 1, 0] = rz
    K[:, 1, 2] = -rx
    K[:, 2, 0] = -ry
    K[:, 2, 1] = rx
    ident = np.eye(3, dtype=np.float64)[None]
    return ident + sin * K + (1.0 - cos) * np.matmul(K, K)


class ManoLayer:
    """Pure-NumPy MANO layer for one handedness ("right" or "left")."""

    def __init__(self, is_rhand: bool = True, model_dir: Optional[str] = None,
                 fix_left_shapedirs: bool = False) -> None:
        self.is_rhand = is_rhand
        model_dir = model_dir or _ASSET_DIR
        fname = "MANO_RIGHT.npz" if is_rhand else "MANO_LEFT.npz"
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MANO model not found: {path}. The release must ship vendored "
                f"MANO_*.npz under assets/mano/.")
        m = np.load(path)
        self.faces = m["f"].astype(np.int64)
        self.v_template = m["v_template"].astype(np.float64)          # (778,3)
        self.shapedirs = m["shapedirs"].astype(np.float64)            # (778,3,10)
        self.posedirs = m["posedirs"].astype(np.float64)              # (778,3,135)
        self.J_regressor = m["J_regressor"].astype(np.float64)        # (16,778)
        self.weights = m["weights"].astype(np.float64)                # (778,16)
        self.kintree_table = m["kintree_table"].astype(np.int64)      # (2,16)
        self.parents = self.kintree_table[0].copy()
        self.parents[0] = -1
        # Optional fix for the well-known left-hand shapedirs sign bug.
        if (not is_rhand) and fix_left_shapedirs:
            self.shapedirs[:, 0, :] *= -1.0

    def _fk(self, rot_mats: np.ndarray, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics. Returns posed joints and relative transforms A.

        Args:
            rot_mats: ``(T, 16, 3, 3)`` local joint rotations.
            J: ``(T, 16, 3)`` rest joint locations.

        Returns:
            (J_transformed ``(T,16,3)``, A ``(T,16,4,4)`` skinning transforms).
        """
        T = rot_mats.shape[0]
        A = np.tile(np.eye(4, dtype=np.float64), (T, 16, 1, 1))
        A[:, :, :3, :3] = rot_mats
        A[:, 0, :3, 3] = J[:, 0]
        for i in range(1, 16):
            p = self.parents[i]
            A[:, i, :3, 3] = J[:, i] - J[:, p]
            A[:, i] = np.matmul(A[:, p], A[:, i])
        J_transformed = A[:, :, :3, 3].copy()
        # Remove the rest-joint translation so LBS leaves the rest pose unchanged.
        zeros = np.zeros((T, 16, 1), dtype=np.float64)
        rest = np.concatenate([J, zeros], axis=2)[..., None]          # (T,16,4,1)
        offset = np.matmul(A, rest)[..., :3, 0]                       # (T,16,3)
        A[:, :, :3, 3] -= offset
        return J_transformed, A

    def forward(self, betas: np.ndarray, global_orient: np.ndarray,
                hand_pose: np.ndarray, transl: np.ndarray,
                return_verts: bool = True) -> Dict[str, np.ndarray]:
        """Run MANO for a sequence of frames.

        Args:
            betas: ``(T, 10)`` shape parameters.
            global_orient: ``(T, 3)`` root rotation (axis-angle).
            hand_pose: ``(T, 45)`` 15-joint pose (axis-angle, absolute).
            transl: ``(T, 3)`` root translation.
            return_verts: also compute the full 778-vertex mesh.

        Returns:
            dict with ``joints`` ``(T,21,3)`` and (optionally) ``vertices``
            ``(T,778,3)`` plus ``faces``.
        """
        betas = np.asarray(betas, dtype=np.float64)
        global_orient = np.asarray(global_orient, dtype=np.float64)
        hand_pose = np.asarray(hand_pose, dtype=np.float64)
        transl = np.asarray(transl, dtype=np.float64)
        T = betas.shape[0]

        v_shaped = self.v_template[None] + np.einsum(
            "bl,mkl->bmk", betas, self.shapedirs)                     # (T,778,3)
        J = np.einsum("bik,ji->bjk", v_shaped, self.J_regressor)      # (T,16,3)

        full_pose = np.concatenate([global_orient, hand_pose], axis=1)  # (T,48)
        rot_mats = _rodrigues(full_pose.reshape(-1, 3)).reshape(T, 16, 3, 3)

        pose_feature = (rot_mats[:, 1:] - np.eye(3)).reshape(T, -1)   # (T,135)
        pose_offsets = np.einsum("bl,mkl->bmk", pose_feature, self.posedirs)
        v_posed = v_shaped + pose_offsets

        J_transformed, A = self._fk(rot_mats, J)

        out: Dict[str, np.ndarray] = {}

        # Decide which vertices to skin: all (mesh) or just fingertips (fast).
        if return_verts:
            verts = self._lbs(v_posed, A, np.arange(v_posed.shape[1]))
            verts = verts + transl[:, None, :]
            out["vertices"] = verts
            tips = verts[:, list(FINGERTIP_VERTEX_IDS.values()), :]
        else:
            tip_ids = list(FINGERTIP_VERTEX_IDS.values())
            tips = self._lbs(v_posed, A, np.asarray(tip_ids)) + transl[:, None, :]

        joints16 = J_transformed + transl[:, None, :]                # (T,16,3)
        stacked = np.concatenate([joints16, tips], axis=1)           # (T,21,3)
        # Reorder [16 joints + 5 tips] -> OpenPose 21 order.
        joints = stacked[:, list(MANO_TO_OPENPOSE), :]
        out["joints"] = joints
        out["faces"] = self.faces
        return out

    def _lbs(self, v_posed: np.ndarray, A: np.ndarray,
             vert_idx: np.ndarray) -> np.ndarray:
        """Linear blend skinning restricted to the requested vertices.

        Args:
            v_posed: ``(T, V, 3)`` posed-shape vertices.
            A: ``(T, 16, 4, 4)`` skinning transforms.
            vert_idx: indices of vertices to skin.

        Returns:
            ``(T, len(vert_idx), 3)`` skinned vertices (before translation).
        """
        T = v_posed.shape[0]
        vp = v_posed[:, vert_idx, :]                                  # (T,V,3)
        w = self.weights[vert_idx]                                    # (V,16)
        V = vp.shape[1]
        vp_homo = np.concatenate([vp, np.ones((T, V, 1))], axis=2)    # (T,V,4)
        out = np.zeros((T, V, 3), dtype=np.float64)
        for j in range(16):
            transformed = np.einsum("tij,tnj->tni", A[:, j], vp_homo)  # (T,V,4)
            out += w[None, :, j, None] * transformed[..., :3]
        return out


# Module-level cache so the (small) model files are loaded once.
_cache: Dict[bool, ManoLayer] = {}


def get_mano_layer(is_rhand: bool = True,
                   fix_left_shapedirs: bool = False) -> ManoLayer:
    """Return a cached :class:`ManoLayer` for the requested handedness."""
    key = bool(is_rhand)
    if key not in _cache:
        _cache[key] = ManoLayer(is_rhand=is_rhand,
                                 fix_left_shapedirs=fix_left_shapedirs)
    return _cache[key]
