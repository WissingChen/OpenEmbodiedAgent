from __future__ import annotations

from pathlib import Path

from PhyAgentOS.config.schema import Config
from hal.hal_watchdog import _resolve_watchdog_topology


def test_watchdog_resolves_robot_workspace_and_shared_environment(monkeypatch, tmp_path: Path) -> None:
    config = Config.model_validate(
        {
            "embodiments": {
                "mode": "fleet",
                "sharedWorkspace": str(tmp_path / "workspaces" / "shared"),
                "instances": [
                    {
                        "robotId": "go2_edu_001",
                        "driver": "go2_edu",
                        "workspace": str(tmp_path / "workspaces" / "go2_edu_001"),
                        "enabled": True,
                    }
                ],
            }
        }
    )
    monkeypatch.setattr("PhyAgentOS.config.loader.load_config", lambda: config)

    robot_workspace, env_file, driver_name, registry = _resolve_watchdog_topology(
        None,
        "simulation",
        "go2_edu_001",
    )

    assert robot_workspace == tmp_path / "workspaces" / "go2_edu_001"
    assert env_file == tmp_path / "workspaces" / "shared" / "ENVIRONMENT.md"
    assert driver_name == "go2_edu"
    assert registry is not None
    assert registry.is_fleet is True
