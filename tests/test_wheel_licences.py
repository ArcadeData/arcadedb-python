"""The licence guard: a wheel that carries a file that is not open source fails the build.

Studio bundled ApexCharts, which is not open source since 5.1.0 (ArcadeData/arcadedb#10579), until engine 26.11.1
replaced it with Apache ECharts (ArcadeData/arcadedb#10627). The tests build tiny wheels in a temporary directory, so
they run in milliseconds and need no built wheel.
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "verify_wheel_licences.py"
)
EXCLUSIONS = Path(__file__).resolve().parents[1] / "scripts" / "jar_exclusions.txt"


def _load():
    spec = importlib.util.spec_from_file_location("verify_wheel_licences", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _jar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_studio_jar_is_not_in_the_exclusion_list():
    # Studio ships from 26.11.1 (ECharts); the guard, not an exclusion, keeps ApexCharts out.
    lines = [ln.strip() for ln in EXCLUSIONS.read_text().splitlines()]
    assert not [ln for ln in lines if not ln.startswith("#") and "studio" in ln]


def test_a_clean_wheel_passes(tmp_path, capsys):
    module = _load()
    wheel = _wheel(
        tmp_path / "a-1-py3-none-any.whl",
        {
            "arcadedb_embedded/jars/arcadedb-server-1.jar": _jar({"A.class": b"x"}),
            "arcadedb_embedded/server.py": b"",
        },
    )
    assert module.main([str(wheel)]) == 0
    assert "OK:" in capsys.readouterr().out


def test_a_studio_jar_with_echarts_passes(tmp_path, capsys):
    module = _load()
    wheel = _wheel(
        tmp_path / "a-1-py3-none-any.whl",
        {
            "arcadedb_embedded/jars/arcadedb-studio-26.11.1.jar": _jar(
                {
                    "static/index.html": b"x",
                    "static/dist/js/echarts.min.js": b"x",
                }
            )
        },
    )
    assert module.main([str(wheel)]) == 0
    assert "OK:" in capsys.readouterr().out


def test_an_echarts_file_outside_a_jar_passes(tmp_path):
    module = _load()
    wheel = _wheel(
        tmp_path / "a-1-py3-none-any.whl",
        {"arcadedb_embedded/static/dist/js/echarts.min.js": b"x"},
    )
    assert module.main([str(wheel)]) == 0


def test_an_apexcharts_file_fails_the_wheel(tmp_path, capsys):
    module = _load()
    wheel = _wheel(
        tmp_path / "a-1-py3-none-any.whl",
        {"arcadedb_embedded/static/dist/js/apexcharts.min.js": b"x"},
    )
    assert module.main([str(wheel)]) == 1
    err = capsys.readouterr().err
    assert "ApexCharts" in err
    assert "apexcharts.min.js" in err


def test_a_studio_jar_from_an_older_engine_fails_the_wheel(tmp_path, capsys):
    module = _load()
    wheel = _wheel(
        tmp_path / "a-1-py3-none-any.whl",
        {
            "arcadedb_embedded/jars/arcadedb-studio-26.10.2.jar": _jar(
                {
                    "static/index.html": b"x",
                    "static/dist/js/apexcharts.min.js": b"x",
                }
            )
        },
    )
    assert module.main([str(wheel)]) == 1
    err = capsys.readouterr().err
    assert "arcadedb-studio-26.10.2.jar contains" in err
    assert "26.11.1" in err


def test_a_renamed_jar_that_carries_apexcharts_fails_the_wheel(tmp_path, capsys):
    module = _load()
    wheel = _wheel(
        tmp_path / "a-1-py3-none-any.whl",
        {
            "arcadedb_embedded/jars/ui-1.jar": _jar(
                {"static/dist/js/ApexCharts.min.js": b"x"}
            )
        },
    )
    assert module.main([str(wheel), str(wheel)]) == 1
    assert "ui-1.jar contains" in capsys.readouterr().err


def test_a_jar_that_is_not_a_zip_fails_the_wheel(tmp_path, capsys):
    module = _load()
    wheel = _wheel(
        tmp_path / "a-1-py3-none-any.whl",
        {"arcadedb_embedded/jars/broken-1.jar": b"not a zip"},
    )
    assert module.main([str(wheel)]) == 1
    assert "cannot be checked" in capsys.readouterr().err


def test_a_missing_file_is_a_usage_error(tmp_path):
    module = _load()
    assert module.main([str(tmp_path / "nope.whl")]) == 2
    assert module.main([]) == 2
