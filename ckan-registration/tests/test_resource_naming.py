"""Resource names (CKAN resource title) must be human-readable and unique."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gam_registration.utils as utils  # noqa: E402


def test_human_readable_name_humanizes_dirs_keeps_filename():
    assert utils.human_readable_resource_name(Path("Model_File/ygjk_tr.dis")) == "Model File / ygjk_tr.dis"
    assert utils.human_readable_resource_name(Path("Report/YGJK_Model_Report.pdf")) == "Report / YGJK_Model_Report.pdf"
    assert utils.human_readable_resource_name(Path("readme.txt")) == "readme.txt"


def test_build_resource_plan_names_are_readable_and_unique(tmp_path):
    root = tmp_path
    (root / "Model_File").mkdir()
    for fn in ("ygjk_tr.dis", "ygjk_tr.nam", "ygjk_tr.bas"):
        (root / "Model_File" / fn).write_text("x")
    (root / "Other_Data").mkdir()
    (root / "Other_Data" / "ygjk_tr.dis").write_text("x")  # same filename, different dir

    files = sorted((root).rglob("*"))
    files = [f for f in files if f.is_file()]
    plan = utils.build_resource_plan(files, root, "http://example.org")

    names = [item["resource_name"] for item in plan]
    assert len(names) == len(set(names)), f"names must be unique: {names}"
    # No machine-y '__' separators; readable ' / ' path + real filename.
    assert all("__" not in n for n in names), names
    assert "Model File / ygjk_tr.dis" in names
    assert "Other Data / ygjk_tr.dis" in names  # disambiguated by folder
