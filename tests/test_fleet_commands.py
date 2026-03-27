from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from OEA.cli.commands import app

runner = CliRunner()


def test_onboard_creates_fleet_shared_and_robot_workspaces(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
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
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("OEA.config.loader.get_config_path", lambda: config_path)

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert (tmp_path / "workspaces" / "shared" / "ROBOTS.md").exists()
    assert (tmp_path / "workspaces" / "shared" / "TASK.md").exists()
    assert (tmp_path / "workspaces" / "shared" / "ORCHESTRATOR.md").exists()
    assert (tmp_path / "workspaces" / "go2_edu_001" / "ACTION.md").exists()
    assert (tmp_path / "workspaces" / "go2_edu_001" / "EMBODIED.md").exists()
