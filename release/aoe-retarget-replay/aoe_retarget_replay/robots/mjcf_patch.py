"""MJCF camera-injection patchers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

EGO_CAMERA_NAME = "phantom_ego"
EXT_CAMERA_NAME = "phantom_external"

_EXT_CAMERA_XML = (
    f'<camera name="{EXT_CAMERA_NAME}" mode="fixed" '
    f'pos="2.2 -1.8 1.4" xyaxes="0.7071 0.7071 0 -0.3 0.3 0.9" fovy="50"/>'
)


def _ego_camera_xml(fovy: float) -> str:
    return (
        f'<camera name="{EGO_CAMERA_NAME}" mode="fixed" '
        f'pos="0 0 1.7" quat="1 0 0 0" fovy="{fovy}"/>'
    )


def _inject_cameras(xml: str, fovy: float) -> str:
    wb_idx = xml.find("<worldbody>")
    if wb_idx < 0:
        raise RuntimeError("No <worldbody> in MJCF")
    insert_at = xml.find(">", wb_idx) + 1
    inject = "\n    " + _ego_camera_xml(fovy) + "\n    " + _EXT_CAMERA_XML
    return xml[:insert_at] + inject + xml[insert_at:]


def patch_mjcf_local(src_xml_path: Path, fovy: float) -> Path:
    """Mirror the MJCF's own directory as symlinks. For Dex3."""
    src_xml_path = Path(src_xml_path).resolve()
    xml = src_xml_path.read_text()
    modified = _inject_cameras(xml, fovy)

    tmpdir = Path(tempfile.mkdtemp(prefix="aoe_mjcf_"))
    src_dir = src_xml_path.parent
    for entry in src_dir.iterdir():
        if entry.is_dir():
            os.symlink(entry, tmpdir / entry.name)
        elif entry.suffix != ".xml":
            os.symlink(entry, tmpdir / entry.name)
    dst = tmpdir / src_xml_path.name
    dst.write_text(modified)
    return dst


def patch_mjcf_with_sibling_dirs(src_xml_path: Path, fovy: float) -> Path:
    """For MJCFs that reference ``../<sibling>/...``. Used by Inspire."""
    src_xml_path = Path(src_xml_path).resolve()
    xml = src_xml_path.read_text()
    modified = _inject_cameras(xml, fovy)

    src_dir = src_xml_path.parent
    parent_dir = src_dir.parent
    tmpdir = Path(tempfile.mkdtemp(prefix="aoe_mjcf_sib_"))
    for sibling in parent_dir.iterdir():
        if sibling.name == src_dir.name:
            continue
        os.symlink(sibling, tmpdir / sibling.name)
    new_src_dir = tmpdir / src_dir.name
    new_src_dir.mkdir()
    for entry in src_dir.iterdir():
        if entry.is_dir():
            os.symlink(entry, new_src_dir / entry.name)
        elif entry.suffix != ".xml":
            os.symlink(entry, new_src_dir / entry.name)
    dst = new_src_dir / src_xml_path.name
    dst.write_text(modified)
    return dst
