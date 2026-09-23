"""ARX5 MjLab entity built from the supplied visualization MJCF."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

SCENE_XML = Path("/home/ymq/code/cvpr2027/assets/mujoco_arx_scene/scene.xml")


def get_arx5_spec() -> mujoco.MjSpec:
    """Return only the ARX body, without table/cameras/debug geometry per env."""
    root = ElementTree.parse(SCENE_XML).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str((SCENE_XML.parent / "meshes").resolve()))
    worldbody = root.find("worldbody")
    assert worldbody is not None
    for child in list(worldbody):
        if child.tag != "body" or child.get("name") != "base_link":
            worldbody.remove(child)
    return mujoco.MjSpec.from_string(ElementTree.tostring(root, encoding="unicode"))


def get_arx5_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_arx5_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
        ),
    )
