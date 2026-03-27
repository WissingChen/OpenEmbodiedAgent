from __future__ import annotations

from pathlib import Path
import textwrap

from hal.drivers import list_drivers, load_driver
from hal.plugins import register_plugin, resolve_external_driver


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def test_external_driver_can_be_registered_and_loaded(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin-home"
    plugin_repo = tmp_path / "sample-plugin"
    monkeypatch.setenv("OEA_PLUGIN_HOME", str(plugin_home))

    _write(
        plugin_repo / "oea_plugin.toml",
        """
        [plugin]
        name = "sample_plugin"
        version = "0.0.1"

        [driver]
        name = "sample_ext"
        module = "sample_plugin.driver"
        class = "SampleDriver"
        profile_path = "sample_plugin/profiles/sample.md"

        [python]
        sys_paths = ["."]
        """,
    )
    _write(
        plugin_repo / "sample_plugin" / "__init__.py",
        """
        from .driver import SampleDriver
        """,
    )
    _write(
        plugin_repo / "sample_plugin" / "driver.py",
        """
        from pathlib import Path

        from hal.base_driver import BaseDriver


        class SampleDriver(BaseDriver):
            def __init__(self, **_kwargs):
                self._scene = {}

            def get_profile_path(self) -> Path:
                return Path(__file__).resolve().parent / "profiles" / "sample.md"

            def load_scene(self, scene: dict[str, dict]) -> None:
                self._scene = dict(scene)

            def execute_action(self, action_type: str, params: dict) -> str:
                return f"{action_type}:{params.get('value', 'ok')}"

            def get_scene(self) -> dict[str, dict]:
                return dict(self._scene)
        """,
    )
    _write(plugin_repo / "sample_plugin" / "profiles" / "sample.md", "# sample\n")

    spec = register_plugin(plugin_repo, source_url=str(plugin_repo), ref="local")
    assert spec.driver_name == "sample_ext"

    resolved = resolve_external_driver("sample_ext")
    assert resolved is not None
    assert resolved.profile_path.exists()

    assert "sample_ext" in list_drivers()
    driver = load_driver("sample_ext")
    driver.load_scene({"cube": {"type": "object"}})

    assert driver.execute_action("ping", {"value": "pong"}) == "ping:pong"
    assert driver.get_scene()["cube"]["type"] == "object"
