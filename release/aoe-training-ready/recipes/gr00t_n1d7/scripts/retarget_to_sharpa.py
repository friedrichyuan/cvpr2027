"""
Retarget MANO 21 keypoints → Sharpa Wave 22-DoF joint angles.

Follows EgoScale methodology:
  - Extract 5 fingertip positions from MANO 21 keypoints
  - Optimize 22 robot joint angles via CasADi + IPOPT to match fingertip positions
  - Subject to joint limit constraints from official URDF
  - Exponential smoothing for temporal coherence

Sharpa Wave 22-DoF joint order (left hand):
  Thumb (5):  CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
  Index (4):  MCP_FE, MCP_AA, PIP, DIP
  Middle (4): MCP_FE, MCP_AA, PIP, DIP
  Ring (4):   MCP_FE, MCP_AA, PIP, DIP
  Pinky (5):  CMC, MCP_FE, MCP_AA, PIP, DIP

Usage:
    cd scripts
    python retarget_to_sharpa.py \
        --data-root /path/to/poc_deliver \
        --keypoints-dir ../output \
        --output-dir ../output \
        --max-episodes 2
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import casadi as ca
import numpy as np
from tqdm import tqdm

from utils import ema_smooth, R_MANO_TO_SHARPA

# ============================================================
# URDF Parsing → Kinematic parameters
# ============================================================

URDF_DIR = Path(__file__).resolve().parent / "urdf" / "sharpa-urdf-usd-xml" / "wave_01"
LEFT_URDF_PATH = URDF_DIR / "left_sharpa_wave" / "left_sharpa_wave.urdf"
RIGHT_URDF_PATH = URDF_DIR / "right_sharpa_wave" / "right_sharpa_wave.urdf"

LEFT_JOINT_NAMES = [
    "left_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA", "left_thumb_IP",
    "left_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP", "left_pinky_DIP",
]

RIGHT_JOINT_NAMES = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
]

LEFT_FINGERTIP_LINKS = [
    "left_thumb_fingertip", "left_index_fingertip",
    "left_middle_fingertip", "left_ring_fingertip", "left_pinky_fingertip",
]

RIGHT_FINGERTIP_LINKS = [
    "right_thumb_fingertip", "right_index_fingertip",
    "right_middle_fingertip", "right_ring_fingertip", "right_pinky_fingertip",
]

# Keypoint indices for fingertips in OpenPose 21-kp ordering
KP_FINGERTIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
# MCP keypoint indices (index, middle, ring, pinky)
KP_MCPS = [5, 9, 13, 17]


def parse_urdf(urdf_path: str) -> dict:
    """Parse URDF to extract joint info: limits, origins, parent-child relationships."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = {}
    for j in root.findall("joint"):
        jtype = j.get("type")
        name = j.get("name")
        origin = j.find("origin")
        xyz = np.array([float(x) for x in origin.get("xyz", "0 0 0").split()])
        rpy = np.array([float(x) for x in origin.get("rpy", "0 0 0").split()])
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")

        limit_el = j.find("limit")
        if jtype == "revolute" and limit_el is not None:
            lo = float(limit_el.get("lower", "0"))
            hi = float(limit_el.get("upper", "0"))
        else:
            lo, hi = 0.0, 0.0

        joints[name] = {
            "type": jtype,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "lower": lo,
            "upper": hi,
        }

    return joints


def _rpy_to_matrix(rpy):
    """Roll-Pitch-Yaw to 3x3 rotation matrix (XYZ extrinsic = ZYX intrinsic)."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ============================================================
# CasADi Symbolic FK
# ============================================================

def _ca_rot_z(theta):
    """CasADi symbolic rotation about Z axis."""
    c = ca.cos(theta)
    s = ca.sin(theta)
    return ca.vertcat(
        ca.horzcat(c, -s, 0),
        ca.horzcat(s, c, 0),
        ca.horzcat(0, 0, 1),
    )


def _ca_transform(R_parent, p_parent, R_joint_origin, p_joint_origin, theta):
    """Compute child frame: T_child = T_parent @ T_joint_origin @ Rz(theta)."""
    R_after_origin = R_parent @ R_joint_origin
    p_after_origin = p_parent + R_parent @ p_joint_origin
    R_child = R_after_origin @ _ca_rot_z(theta)
    return R_child, p_after_origin


def build_fk_casadi(urdf_joints: dict, joint_names: list, side: str):
    """
    Build CasADi symbolic function that computes fingertip AND MCP positions from 22 joint angles.

    Returns:
        fk_fn: CasADi Function(q[22]) → fingertips[15] (5 tips × 3D, flattened)
        fk_mcp_fn: CasADi Function(q[22]) → mcp_positions[12] (4 MCPs × 3D, flattened, index/middle/ring/pinky)
        joint_limits: (22, 2) array of [lower, upper]
    """
    q = ca.SX.sym("q", 22)

    joint_limits = np.zeros((22, 2))
    for i, jname in enumerate(joint_names):
        jinfo = urdf_joints[jname]
        joint_limits[i] = [jinfo["lower"], jinfo["upper"]]

    def _get_chain(prefix, joint_indices, stop_at_joint=None):
        """Trace FK chain for a finger, return position (CasADi expr).
        If stop_at_joint is set, return position at that joint index instead of fingertip."""
        R = ca.SX.eye(3)
        p = ca.SX.zeros(3)

        for ji in joint_indices:
            jname = joint_names[ji]
            jinfo = urdf_joints[jname]
            xyz = jinfo["xyz"]
            rpy = jinfo["rpy"]
            R_origin = _rpy_to_matrix(rpy).astype(np.float64)
            p_origin = ca.DM(xyz)
            R, p = _ca_transform(R, p, ca.DM(R_origin), p_origin, q[ji])
            if stop_at_joint is not None and ji == stop_at_joint:
                return p

        last_jname = joint_names[joint_indices[-1]]
        last_child = urdf_joints[last_jname]["child"]

        current_link = last_child
        target_tip = prefix + "_fingertip"

        for _ in range(10):
            found = False
            for jname_f, jinfo_f in urdf_joints.items():
                if jinfo_f["type"] == "fixed" and jinfo_f["parent"] == current_link:
                    xyz_f = jinfo_f["xyz"]
                    rpy_f = jinfo_f["rpy"]
                    R_f = _rpy_to_matrix(rpy_f).astype(np.float64)
                    p = p + R @ ca.DM(xyz_f)
                    R = R @ ca.DM(R_f)
                    current_link = jinfo_f["child"]
                    found = True
                    if current_link == target_tip:
                        return p
                    break
            if not found:
                break

        return p

    # Finger chains (indices into the 22-joint vector)
    thumb_tip = _get_chain(f"{side}_thumb", [0, 1, 2, 3, 4])
    index_tip = _get_chain(f"{side}_index", [5, 6, 7, 8])
    middle_tip = _get_chain(f"{side}_middle", [9, 10, 11, 12])
    ring_tip = _get_chain(f"{side}_ring", [13, 14, 15, 16])
    pinky_tip = _get_chain(f"{side}_pinky", [17, 18, 19, 20, 21])

    tips = ca.vertcat(thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip)
    fk_fn = ca.Function(f"sharpa_fk_{side}", [q], [tips], ["q"], ["tips"])

    # MCP positions (first joint of index/middle/ring/pinky = where the finger starts)
    index_mcp = _get_chain(f"{side}_index", [5, 6, 7, 8], stop_at_joint=5)
    middle_mcp = _get_chain(f"{side}_middle", [9, 10, 11, 12], stop_at_joint=9)
    ring_mcp = _get_chain(f"{side}_ring", [13, 14, 15, 16], stop_at_joint=13)
    pinky_mcp = _get_chain(f"{side}_pinky", [17, 18, 19, 20, 21], stop_at_joint=18)

    mcps = ca.vertcat(index_mcp, middle_mcp, ring_mcp, pinky_mcp)
    fk_mcp_fn = ca.Function(f"sharpa_fk_mcp_{side}", [q], [mcps], ["q"], ["mcps"])

    return fk_fn, fk_mcp_fn, joint_limits


# ============================================================
# Retargeting with IPOPT
# ============================================================

class SharpaRetargeter:
    """CasADi+IPOPT based retargeting from fingertip targets to 22 joint angles."""

    def __init__(self, side: str = "left", scale: float = None):
        if side == "left":
            urdf_path = str(LEFT_URDF_PATH)
            joint_names = LEFT_JOINT_NAMES
        else:
            urdf_path = str(RIGHT_URDF_PATH)
            joint_names = RIGHT_JOINT_NAMES

        urdf_joints = parse_urdf(urdf_path)
        self.fk_fn, self.fk_mcp_fn, self.joint_limits = build_fk_casadi(urdf_joints, joint_names, side)

        # Sharpa middle finger length at rest (palm root → middle fingertip)
        q_zero = np.zeros(22)
        tips_zero = np.array(self.fk_fn(q_zero)).flatten()
        self.sharpa_middle_len = np.linalg.norm(tips_zero[6:9])
        self.fixed_scale = scale

        self._build_solver()

    def compute_scale(self, keypoints_21):
        """Scale factor. Sharpa Wave is designed as 1:1 human hand replica."""
        if self.fixed_scale is not None:
            return self.fixed_scale
        return 1.0

    def _build_solver(self):
        """Build CasADi NLP solver for 5 fingertip targets."""
        q = ca.SX.sym("q", 22)
        target = ca.SX.sym("target", 15)  # 5 tips × 3D

        tips = self.fk_fn(q)
        residual = tips - target
        objective = ca.dot(residual, residual)

        nlp = {"x": q, "f": objective, "p": target}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 150,
            "ipopt.tol": 1e-6,
            "print_time": 0,
            "ipopt.warm_start_init_point": "yes",
        }
        self.solver = ca.nlpsol("retarget", "ipopt", nlp, opts)
        self.lbx = self.joint_limits[:, 0].tolist()
        self.ubx = self.joint_limits[:, 1].tolist()

    def retarget_frame(self, target_tips: np.ndarray, scale: float = 1.0,
                       q_init: np.ndarray = None) -> np.ndarray:
        """
        Solve for 22 joint angles matching 5 target fingertip positions.

        Args:
            target_tips: (5, 3) fingertip positions in hand-local frame (human scale)
            scale: human→Sharpa scale factor
            q_init: (22,) initial guess

        Returns:
            q_opt: (22,) optimized joint angles
        """
        scaled_target = (target_tips * scale).flatten()

        if q_init is None:
            q_init = (np.array(self.lbx) + np.array(self.ubx)) / 2

        sol = self.solver(
            x0=q_init,
            lbx=self.lbx,
            ubx=self.ubx,
            p=scaled_target,
        )
        return np.array(sol["x"]).flatten()


# ============================================================
# Batch processing
# ============================================================


def _build_hand_frame(keypoints_t, side="left"):
    """
    Build a local coordinate frame from 21 keypoints at one timestep,
    aligned with Sharpa URDF convention:
      Z = finger direction (wrist → middle MCP)
      Y = side direction (toward pinky for left hand, toward thumb for right hand)
      X = palm normal (right-hand rule)

    Args:
        keypoints_t: (21, 3) keypoints at one frame
        side: "left" or "right"

    Returns R (3x3) where columns = [X_palm, Y_side, Z_finger] in the input frame.
    R.T maps input-frame vectors to Sharpa-local vectors.
    """
    wrist = keypoints_t[0]
    middle_mcp = keypoints_t[9]
    index_mcp = keypoints_t[5]
    ring_mcp = keypoints_t[13]

    # Z axis: finger direction (wrist → middle MCP)
    z_axis = middle_mcp - wrist
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return np.eye(3)
    z_axis = z_axis / z_norm

    # Y axis: side direction
    # Left hand Sharpa: +Y = pinky side → ring_MCP - index_MCP
    # Right hand Sharpa: +Y = thumb side → index_MCP - ring_MCP
    if side == "left":
        side_vec = ring_mcp - index_mcp
    else:
        side_vec = index_mcp - ring_mcp

    # Orthogonalize against Z
    y_axis = side_vec - np.dot(side_vec, z_axis) * z_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return np.eye(3)
    y_axis = y_axis / y_norm

    # X axis: palm normal (right-hand rule: Y × Z)
    x_axis = np.cross(y_axis, z_axis)

    # R: columns = [X_palm, Y_side, Z_finger]
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


def retarget_sequence(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    retargeter: SharpaRetargeter,
    side: str = "left",
    ema_alpha: float = 0.3,
) -> np.ndarray:
    """
    Retarget a full sequence of 21 keypoints to Sharpa 22-DoF joints.

    Uses per-episode adaptive scale (median of all valid frames) for stability.
    """
    T = len(keypoints)
    joints = np.zeros((T, 22), dtype=np.float64)
    q_prev = None

    # Compute stable scale: median over all valid frames
    if retargeter.fixed_scale is not None:
        scale = retargeter.fixed_scale
    else:
        scales = []
        for t in range(T):
            if valid_mask[t]:
                s = retargeter.compute_scale(keypoints[t])
                if 0.5 < s < 3.0:
                    scales.append(s)
        scale = float(np.median(scales)) if scales else 1.13

    for t in range(T):
        if not valid_mask[t]:
            if q_prev is not None:
                joints[t] = q_prev
            continue

        wrist = keypoints[t, 0]
        tips_world = keypoints[t, KP_FINGERTIPS] - wrist

        R_hand = _build_hand_frame(keypoints[t], side=side)
        tips_local = (R_hand.T @ tips_world.T).T

        try:
            q_opt = retargeter.retarget_frame(tips_local, scale=scale, q_init=q_prev)
            joints[t] = q_opt
            q_prev = q_opt
        except Exception:
            if q_prev is not None:
                joints[t] = q_prev

    # EMA smoothing on valid frames
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 1:
        joints[valid_indices] = ema_smooth(joints[valid_indices], alpha=ema_alpha)

    return joints.astype(np.float32)


def convert_single(
    keypoints_path: str, output_path: str,
    left_retargeter: SharpaRetargeter, right_retargeter: SharpaRetargeter,
    ema_alpha: float = 0.3,
) -> dict:
    """Convert one hands_keypoints.npz → hands_retargeted_sharpa.npz."""
    data = np.load(keypoints_path)
    joints_world = data["joints_world"]  # (2, T, 21, 3)
    pred_valid = data["pred_valid"]      # (2, T)

    left_valid = pred_valid[0] > 0.5
    right_valid = pred_valid[1] > 0.5

    left_joints = retarget_sequence(joints_world[0], left_valid, left_retargeter, side="left", ema_alpha=ema_alpha)
    right_joints = retarget_sequence(joints_world[1], right_valid, right_retargeter, side="right", ema_alpha=ema_alpha)

    np.savez_compressed(
        output_path,
        left_hand_joints=left_joints,    # (T, 22)
        right_hand_joints=right_joints,  # (T, 22)
        pred_valid=pred_valid,
    )

    T = joints_world.shape[1]
    both_valid = left_valid & right_valid
    return {"frames": T, "valid_ratio": float(both_valid.sum() / T)}


def main():
    parser = argparse.ArgumentParser(
        description="Retarget MANO 21 keypoints → Sharpa Wave 22-DoF joints (CasADi+IPOPT)"
    )
    parser.add_argument("--data-root", type=str,
                        default="/media/hdd4tb/sankuai/code/07_具身/dataset/07_Ego/poc_deliver")
    parser.add_argument("--keypoints-dir", type=str, default=None,
                        help="Directory containing <episode>/hands_keypoints.npz")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Writes <output-dir>/<episode>/hands_retargeted_sharpa.npz")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=None,
                        help="Manual scale factor (auto-computed from URDF if not set)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.keypoints_dir:
        kp_root = Path(args.keypoints_dir)
        episodes = sorted([
            d for d in data_root.iterdir()
            if d.is_dir()
            and (kp_root / d.name / "hands_keypoints.npz").exists()
        ])
    else:
        episodes = sorted([
            d for d in data_root.iterdir()
            if d.is_dir()
            and (d / "ego_process" / "ego_hands_reconstruction" / "hands_keypoints.npz").exists()
        ])

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"=== Retarget → Sharpa Wave (22-DoF, CasADi+IPOPT) ===")
    print(f"  Episodes: {len(episodes)}")
    print(f"  EMA alpha: {args.ema_alpha}")
    print(f"  Left URDF: {LEFT_URDF_PATH}")
    print(f"  Right URDF: {RIGHT_URDF_PATH}")

    left_retargeter = SharpaRetargeter(side="left", scale=args.scale)
    right_retargeter = SharpaRetargeter(side="right", scale=args.scale)
    scale_mode = f"fixed={args.scale}" if args.scale else "adaptive (per-frame)"
    print(f"  Scale: {scale_mode}")
    print(f"  Sharpa middle finger: {left_retargeter.sharpa_middle_len*1000:.1f} mm")
    print()

    success = 0
    skipped = 0
    failed = []

    for ep_dir in tqdm(episodes, desc="Retargeting"):
        if args.keypoints_dir:
            kp_path = Path(args.keypoints_dir) / ep_dir.name / "hands_keypoints.npz"
        else:
            kp_path = ep_dir / "ego_process" / "ego_hands_reconstruction" / "hands_keypoints.npz"

        if args.output_dir:
            out_path = Path(args.output_dir) / ep_dir.name / "hands_retargeted_sharpa.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = kp_path.parent / "hands_retargeted_sharpa.npz"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            stats = convert_single(str(kp_path), str(out_path), left_retargeter, right_retargeter, ema_alpha=args.ema_alpha)
            success += 1
        except Exception as e:
            failed.append(f"{ep_dir.name}: {e}")

    print(f"\nDone: {success} converted, {skipped} skipped, {len(failed)} failed")
    if failed:
        for f in failed[:10]:
            print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
