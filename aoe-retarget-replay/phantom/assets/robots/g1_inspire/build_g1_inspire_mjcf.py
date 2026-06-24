# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python3
"""Build a G1 + Inspire-hand MJCF by surgery on the existing Dex3 MJCF.

Source files (read-only):
  - G1 MJCF with Dex3 hands (in-tree, self-contained):
      assets/robots/g1_dex3/g1_mocap_29dof_with_dex3_hands.xml
  - Merged G1 + Inspire URDF (for hand kinematic + mesh refs; one-shot
    external reference, only needed at build time):
      /home/yifan/projects/humanoid-dexart-retarget/assets/g1_with_inspire_hand/
          g1_29dof_rev_1_0_with_inspire_hands.urdf

Output:
  assets/robots/g1_inspire/g1_mocap_29dof_with_inspire_hands.xml

Surgery steps:
  1) Strip Dex3 hand sub-trees and Dex3 mesh declarations.
  2) Append Inspire hand sub-trees under {side}_wrist_yaw_link.
  3) Add Inspire mesh declarations (collision .obj reused as visual; .glb is
     unsupported by MuJoCo).
  4) Add 12 equality constraints implementing the Inspire URDF mimic rules
     (6 per hand).
  5) Rewrite <actuator>: drop Dex3 hand actuators (14), add Inspire hand
     actuators (12).
  6) Drop the original <keyframe>; downstream pipeline always initialises
     qpos manually, and the original keyframe has the wrong width.

Mesh strategy (self-contained, no symlinks):
  - Set meshdir="." (relative to the output XML).
  - Rewrite Dex3 STL refs as <mesh file="../g1_base/meshes/foo.STL"/> so the
    output XML uses in-tree shared base meshes.
  - Inspire meshes live in ./meshes/*.obj (copied separately).
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

# ── Constants ──────────────────────────────────────────────────────────────

ASSET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ASSET_DIR.parent.parent.parent  # .../egodex_to_g1
SRC_MJCF = PROJECT_ROOT / "assets" / "robots" / "g1_dex3" / "g1_mocap_29dof_with_dex3_hands.xml"
SRC_URDF = Path(
    "/home/yifan/projects/humanoid-dexart-retarget/assets/g1_with_inspire_hand/"
    "g1_29dof_rev_1_0_with_inspire_hands.urdf"
)
OUT_MJCF = ASSET_DIR / "g1_mocap_29dof_with_inspire_hands.xml"
# Inspire-specific finger meshes live in ./meshes/ next to the output XML.
INSPIRE_MESH_DIR = ASSET_DIR / "meshes"
# Relative path (from the output XML's location) to the shared G1 base meshes.
BASE_MESHES_REL = "../g1_base/meshes"

# Dex3 finger root bodies under each side's wrist_yaw_link.
DEX3_FINGER_ROOTS = ("hand_thumb_0_link", "hand_middle_0_link", "hand_index_0_link")

# Dex3 hand mesh names to strip from <asset>. Palm and wrist_yaw meshes stay
# (palm is the hardware mount surface; Inspire bolts on outside it).
DEX3_MESH_NAMES = {
    f"{side}_hand_{name}"
    for side in ("left", "right")
    for name in ("thumb_0_link", "thumb_1_link", "thumb_2_link",
                 "middle_0_link", "middle_1_link",
                 "index_0_link", "index_1_link")
}

# Dex3 hand actuator names to strip from <actuator>.
DEX3_ACTUATOR_NAMES = {
    f"{side}_hand_{name}"
    for side in ("left", "right")
    for name in ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
                 "middle_0_joint", "middle_1_joint",
                 "index_0_joint", "index_1_joint")
}

# Per-side mount transform (parsed from the merged URDF's mount joint origin).
#   Left mount  : rpy=(0, 0, 1.5708) ; xyz=(0,0,0)
#   Right mount : rpy=(0, 3.1415, 1.5708) ; xyz=(0,0,0)
MOUNT_RPY = {
    "left": (0.0, 0.0, 1.5708),
    "right": (0.0, 3.1415, 1.5708),
}

# Inspire mimic rules: (mimic_joint, driver_joint, multiplier, offset).
# Same coefficients for both sides; we add the side prefix in code.
INSPIRE_MIMIC_RULES_PER_SIDE = [
    ("thumb_intermediate_joint",  "thumb_proximal_pitch_joint", 1.334,    0.0),
    ("thumb_distal_joint",        "thumb_proximal_pitch_joint", 0.667,    0.0),
    ("index_intermediate_joint",  "index_proximal_joint",       1.06399, -0.04545),
    ("middle_intermediate_joint", "middle_proximal_joint",      1.06399, -0.04545),
    ("ring_intermediate_joint",   "ring_proximal_joint",        1.06399, -0.04545),
    ("pinky_intermediate_joint",  "pinky_proximal_joint",       1.06399, -0.04545),
]

# Inspire actuated joints (per side, in canonical action-vector order).
INSPIRE_ACTUATED_PER_SIDE = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]

# ── URDF → MJCF helpers ────────────────────────────────────────────────────


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF rpy (extrinsic XYZ = intrinsic ZYX) → MuJoCo quat (w,x,y,z).

    URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qw, qx, qy, qz


def fmt_floats(*vals: float) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def parse_xyz(elem: ET.Element | None, attr: str = "xyz") -> tuple[float, float, float]:
    if elem is None:
        return (0.0, 0.0, 0.0)
    s = elem.attrib.get(attr, "0 0 0")
    parts = [float(x) for x in s.split()]
    while len(parts) < 3:
        parts.append(0.0)
    return tuple(parts[:3])  # type: ignore[return-value]


def parse_origin(joint_or_link: ET.Element) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ((xyz), (rpy)) from a URDF <origin>; defaults to zeros."""
    origin = joint_or_link.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return parse_xyz(origin, "xyz"), parse_xyz(origin, "rpy")


def parse_axis(joint: ET.Element) -> tuple[float, float, float]:
    axis = joint.find("axis")
    return parse_xyz(axis, "xyz") if axis is not None else (1.0, 0.0, 0.0)


def parse_limits(joint: ET.Element) -> tuple[float, float] | None:
    lim = joint.find("limit")
    if lim is None:
        return None
    return float(lim.attrib["lower"]), float(lim.attrib["upper"])


def parse_inertial(link: ET.Element) -> dict[str, str] | None:
    """Return attrs for MJCF <inertial> from URDF <inertial>; None if absent.

    MuJoCo disallows specifying both ``fullinertia`` and any orientation
    (``quat``/``axisangle``/...): a fullinertia matrix is by definition
    expressed in the body frame, while ``diaginertia`` is a principal-axis
    description that needs a quat to give those axes. If the URDF inertial
    has a non-identity ``rpy``, we first rotate the inertia tensor into the
    body (zero-rotation) frame: ``I_body = R · I_inertial · Rᵀ``.
    """
    import numpy as np

    inertial = link.find("inertial")
    if inertial is None:
        return None
    xyz, rpy = parse_origin(inertial)
    mass = float(inertial.find("mass").attrib["value"])  # type: ignore[union-attr]
    i = inertial.find("inertia")
    if i is None:
        return None
    ixx = float(i.attrib["ixx"]); iyy = float(i.attrib["iyy"]); izz = float(i.attrib["izz"])
    ixy = float(i.attrib["ixy"]); ixz = float(i.attrib["ixz"]); iyz = float(i.attrib["iyz"])
    I = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ], dtype=np.float64)

    # Rotate into body frame if rpy is non-zero.
    if any(abs(r) > 1e-12 for r in rpy):
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        I = R @ I @ R.T

    attrs = {
        "pos": fmt_floats(*xyz),
        "mass": f"{mass:.6g}",
        "fullinertia": fmt_floats(I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]),
    }
    return attrs


def parse_collision_mesh(link: ET.Element) -> str | None:
    """Return the .obj basename (no extension) of the *first* collision mesh,
    or None if the link only has primitive collisions / none at all.
    """
    for coll in link.findall("collision"):
        geom = coll.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        fn = mesh.attrib.get("filename", "")
        return Path(fn).stem
    return None


# ── URDF parsing ───────────────────────────────────────────────────────────


def load_urdf_hand_data(urdf_path: Path) -> dict:
    """Extract per-link / per-joint data needed to emit MJCF hand subtrees.

    Returns a dict keyed by side ("left"/"right") with sub-keys:
      - links: {link_name: {"inertial": <mjcf-attrs|None>, "collision_mesh": stem|None}}
      - joints: ordered list of {name, parent, child, type, xyz, rpy, axis, limits}
        — only joints whose parent or child belongs to a hand subtree.
      - tip_joints: list of {parent, child, xyz, rpy} (fixed) — they define
        leaf bodies used as fingertip targets by dex-retargeting.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Inventory all links and joints.
    all_links = {l.attrib["name"]: l for l in root.findall("link")}
    all_joints = list(root.findall("joint"))

    out: dict[str, dict] = {}
    for side, prefix in [("left", "l_"), ("right", "r_")]:
        side_link_names = [n for n in all_links if n.startswith(prefix)]
        link_data: dict[str, dict] = {}
        for ln in side_link_names:
            link_data[ln] = {
                "inertial": parse_inertial(all_links[ln]),
                "collision_mesh": parse_collision_mesh(all_links[ln]),
            }

        # Joints whose child is a hand link (or whose name starts with the
        # side prefix). We sort by URDF document order, which already gives
        # the natural depth-first traversal.
        joint_data: list[dict] = []
        tip_data: list[dict] = []
        for j in all_joints:
            name = j.attrib.get("name", "")
            child_el = j.find("child")
            parent_el = j.find("parent")
            if child_el is None or parent_el is None:
                continue
            child = child_el.attrib["link"]
            parent = parent_el.attrib["link"]
            if not (child.startswith(prefix) or parent.startswith(prefix)
                    or name.startswith(prefix) or name == f"{prefix}lgripper_mount"
                    or name == "rgripper_mount"):
                continue
            jtype = j.attrib.get("type", "revolute")
            xyz, rpy = parse_origin(j)
            entry = {
                "name": name,
                "parent": parent,
                "child": child,
                "type": jtype,
                "xyz": xyz,
                "rpy": rpy,
            }
            if jtype == "fixed":
                if child.endswith("_tip"):
                    tip_data.append(entry)
                # Skip non-tip fixed joints — only the mount joint is the
                # other fixed joint and it's handled separately.
            else:
                entry["axis"] = parse_axis(j)
                entry["limits"] = parse_limits(j)
                joint_data.append(entry)

        out[side] = {
            "links": link_data,
            "joints": joint_data,
            "tip_joints": tip_data,
        }
    return out


# ── MJCF emission ──────────────────────────────────────────────────────────


def _set(elem: ET.Element, **kwargs: str) -> None:
    for k, v in kwargs.items():
        elem.set(k, v)


def make_hand_subtree(side: str, hand_data: dict) -> ET.Element:
    """Construct the MJCF <body name="{side}_hand_base"> sub-tree.

    Layout (child-of-{side}_wrist_yaw_link, attached via the mount origin):
        body {side}_hand_base
            inertial (from URDF l_hand_base_link / r_hand_base_link)
            geom (collision, primitive boxes/cylinders from URDF — skipped
                  for simplicity since we don't need accurate hand-base
                  collision for retargeting)
            body {side}_thumb_proximal_base   (yaw joint)
                body {side}_thumb_proximal    (pitch joint)
                    body {side}_thumb_intermediate (intermediate joint, mimic)
                        body {side}_thumb_distal   (distal joint, mimic)
                            body {side}_thumb_tip  (fixed)
            body {side}_index_proximal        (proximal joint)
                body {side}_index_intermediate
                    body {side}_index_tip
            ... (middle, ring, pinky)

    Body / joint names use the side prefix `l_`/`r_` from the URDF, but we
    REWRITE them to `{side}_inspire_*` to keep MJCF joint names unambiguous
    and grep-friendly. Wait — actually the URDF already uses `l_`/`r_`
    prefixes (e.g. `l_thumb_proximal_yaw_joint`) which are sufficiently
    distinctive. Keep them.
    """
    prefix = "l_" if side == "left" else "r_"
    links = hand_data["links"]
    joints = hand_data["joints"]
    tip_joints = hand_data["tip_joints"]

    # Index joints by child link.
    joint_by_child: dict[str, dict] = {}
    for j in joints:
        joint_by_child[j["child"]] = j
    tip_by_child: dict[str, dict] = {}
    for t in tip_joints:
        tip_by_child[t["child"]] = t

    # Mount transform (parent=wrist_yaw_link, child=hand_base_link).
    mount_rpy = MOUNT_RPY[side]
    qw, qx, qy, qz = rpy_to_quat_wxyz(*mount_rpy)
    hand_base_link_name = f"{prefix}hand_base_link"
    root_body = ET.Element("body", {
        "name": hand_base_link_name,
        "pos": fmt_floats(0.0, 0.0, 0.0),
        "quat": fmt_floats(qw, qx, qy, qz),
    })
    inertial_attrs = links[hand_base_link_name]["inertial"]
    if inertial_attrs:
        ET.SubElement(root_body, "inertial", inertial_attrs)

    def emit_link_body(parent_xml: ET.Element, link_name: str) -> ET.Element:
        """Emit a body for link_name as a child of parent_xml.

        Looks up the joint whose child==link_name to build pos/quat and the
        joint element. Recurses into all child links via joint_by_child.
        """
        j = joint_by_child[link_name]
        xyz = j["xyz"]; rpy = j["rpy"]; axis = j["axis"]
        qw, qx, qy, qz = rpy_to_quat_wxyz(*rpy)
        body = ET.SubElement(parent_xml, "body", {
            "name": link_name,
            "pos": fmt_floats(*xyz),
            "quat": fmt_floats(qw, qx, qy, qz),
        })
        # inertial
        if links[link_name]["inertial"]:
            ET.SubElement(body, "inertial", links[link_name]["inertial"])
        # joint
        joint_attrs = {
            "name": j["name"],
            "axis": fmt_floats(*axis),
        }
        if j["limits"] is not None:
            lo, hi = j["limits"]
            joint_attrs["range"] = fmt_floats(lo, hi)
        ET.SubElement(body, "joint", joint_attrs)
        # geoms (use collision mesh as both visual and collision)
        mesh_stem = links[link_name]["collision_mesh"]
        if mesh_stem is not None:
            ET.SubElement(body, "geom", {
                "class": "visual",
                "type": "mesh",
                "mesh": mesh_stem,
            })
            ET.SubElement(body, "geom", {
                "class": "collision",
                "type": "mesh",
                "mesh": mesh_stem,
            })
        # Recurse: find children whose parent==link_name.
        for child_link, jj in joint_by_child.items():
            if jj["parent"] == link_name:
                emit_link_body(body, child_link)
        # Attach fixed tip bodies (e.g. l_thumb_tip).
        for child_link, tj in tip_by_child.items():
            if tj["parent"] == link_name:
                tip_xyz = tj["xyz"]; tip_rpy = tj["rpy"]
                tqw, tqx, tqy, tqz = rpy_to_quat_wxyz(*tip_rpy)
                ET.SubElement(body, "body", {
                    "name": child_link,
                    "pos": fmt_floats(*tip_xyz),
                    "quat": fmt_floats(tqw, tqx, tqy, tqz),
                })
        return body

    # Recurse from the hand base.
    for child_link, jj in joint_by_child.items():
        if jj["parent"] == hand_base_link_name:
            emit_link_body(root_body, child_link)

    return root_body


def get_inspire_mesh_names() -> set[str]:
    """All Inspire mesh names (basenames without extension) that need <mesh>
    entries in the MJCF <asset> section. Includes both sides' meshes; some
    are referenced from multiple links (e.g. index_proximal.obj is reused
    by middle/ring/pinky proximal links per the URDF).
    """
    names: set[str] = set()
    for f in INSPIRE_MESH_DIR.glob("*.obj"):
        names.add(f.stem)
    return names


# ── Main surgery ───────────────────────────────────────────────────────────


def find_subtree_by_name(parent: ET.Element, name: str) -> ET.Element | None:
    """Locate the FIRST <body name="..."> descendant of parent."""
    if parent.attrib.get("name") == name:
        return parent
    for child in parent:
        if child.tag != "body":
            continue
        hit = find_subtree_by_name(child, name)
        if hit is not None:
            return hit
    return None


def remove_subtree(root: ET.Element, name: str) -> bool:
    """Remove the <body name="..."> descendant of root (in-place)."""
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.attrib.get("name") == name:
                parent.remove(child)
                return True
    return False


def main() -> None:
    # ── Pre-flight: ensure Inspire mesh dir has been populated ──
    if not INSPIRE_MESH_DIR.exists() or not any(INSPIRE_MESH_DIR.iterdir()):
        raise FileNotFoundError(
            f"Inspire mesh directory empty or missing: {INSPIRE_MESH_DIR}.\n"
            "Run: cp /home/yifan/projects/humanoid-dexart-retarget/assets/"
            "g1_with_inspire_hand/inspire_hands/meshes/collision/*.obj "
            f"{INSPIRE_MESH_DIR}/"
        )

    # ── Parse sources ──
    print(f"[..] Parsing {SRC_MJCF}")
    tree = ET.parse(SRC_MJCF)
    root = tree.getroot()
    print(f"[..] Parsing {SRC_URDF}")
    hand_data = load_urdf_hand_data(SRC_URDF)

    # ── Update <compiler> to point meshdir at this folder (relative ".") ──
    # Inspire .obj files live in ./meshes/ next to the output XML;
    # shared G1 base STLs are referenced via ../g1_base/meshes/ so the result
    # is self-contained without any symlinks.
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", ".")
    # The source Dex3 MJCF uses meshdir="../g1_base/meshes" with bare
    # filenames; flatten those to explicit ../g1_base/meshes/foo.STL so they
    # resolve under the new meshdir=".".
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            fn = mesh.attrib.get("file", "")
            if not fn:
                continue
            if "/" not in fn:
                mesh.set("file", f"{BASE_MESHES_REL}/{fn}")

    # ── 1) Strip Dex3 finger subtrees ──
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for side in ("left", "right"):
        for root_name in DEX3_FINGER_ROOTS:
            full = f"{side}_{root_name}"
            ok = remove_subtree(worldbody, full)
            assert ok, f"Failed to strip Dex3 subtree {full}"
    print("[OK] Removed Dex3 finger subtrees.")

    # ── 2) Strip Dex3 mesh entries ──
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            file_attr = mesh.attrib.get("file", "")
            stem = Path(file_attr).stem
            name_attr = mesh.attrib.get("name", stem)
            if name_attr in DEX3_MESH_NAMES:
                asset.remove(mesh)
    print("[OK] Removed Dex3 mesh entries.")

    # ── 3) Add Inspire mesh entries ──
    inspire_mesh_names = get_inspire_mesh_names()
    assets = root.findall("asset")
    main_asset = assets[0]  # use the first <asset>
    for name in sorted(inspire_mesh_names):
        ET.SubElement(main_asset, "mesh", {
            "name": name,
            "file": f"meshes/{name}.obj",
        })
    print(f"[OK] Added {len(inspire_mesh_names)} Inspire mesh entries.")

    # ── 4) Append Inspire hand subtrees under each wrist_yaw_link ──
    for side in ("left", "right"):
        parent_link_name = f"{side}_wrist_yaw_link"
        parent_body = find_subtree_by_name(worldbody, parent_link_name)
        assert parent_body is not None, f"{parent_link_name} not found"
        subtree = make_hand_subtree(side, hand_data[side])
        parent_body.append(subtree)
    print("[OK] Appended Inspire hand subtrees.")

    # ── 5) Add equality constraints for mimic joints ──
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for side in ("left", "right"):
        prefix = "l_" if side == "left" else "r_"
        for mimic_short, driver_short, mult, off in INSPIRE_MIMIC_RULES_PER_SIDE:
            ET.SubElement(equality, "joint", {
                "joint1": f"{prefix}{mimic_short}",
                "joint2": f"{prefix}{driver_short}",
                # polycoef: A = c0 + c1*B + c2*B^2 + ...
                "polycoef": fmt_floats(off, mult, 0, 0, 0),
            })
    print(f"[OK] Added 12 mimic equality constraints.")

    # ── 6) Rewrite <actuator>: drop Dex3, add Inspire ──
    actuator = root.find("actuator")
    assert actuator is not None
    for act in list(actuator.findall("position")):
        if act.attrib.get("name") in DEX3_ACTUATOR_NAMES:
            actuator.remove(act)
    for side in ("left", "right"):
        prefix = "l_" if side == "left" else "r_"
        for short in INSPIRE_ACTUATED_PER_SIDE:
            full = f"{prefix}{short}"
            ET.SubElement(actuator, "position", {
                "class": "g1",
                "name": full,
                "joint": full,
            })
    print("[OK] Rewrote <actuator>.")

    # ── 7) Drop keyframe (incompatible widths after hand swap) ──
    for kf in list(root.findall("keyframe")):
        root.remove(kf)
    print("[OK] Removed keyframe (will be re-initialised at runtime).")

    # ── 8) Update model name ──
    root.set("model", "g1_29dof_with_inspire_hands_scene")

    # ── Write output ──
    # ET pretty-printing was added in 3.9; we manually use indent if available.
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(OUT_MJCF, encoding="utf-8", xml_declaration=False)
    print(f"[OK] Wrote {OUT_MJCF}")


if __name__ == "__main__":
    main()
