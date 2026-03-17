"""
hal/simulation/scene_io.py

Reads and writes the ENVIRONMENT.md scene-graph.

The scene is represented as a flat JSON object:

    {
      "red_apple": {
        "type": "fruit",
        "color": "red",
        "position": {"x": 5, "y": 5, "z": 0},
        "location": "table"
      },
      ...
    }

ENVIRONMENT.md stores this JSON inside a fenced code block so that both
humans and the LLM agent can read it easily.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── Markdown fences kept as constants (avoids backtick confusion in f-strings) ──
_FENCE_OPEN = "```json"
_FENCE_CLOSE = "```"

_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def load_scene_from_md(path: Path) -> dict[str, dict]:
    """Return the scene dict parsed from *path* (ENVIRONMENT.md).

    Returns an empty dict if the file does not exist or contains no
    valid JSON code block.
    """
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    match = _BLOCK_RE.search(content)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_scene_to_md(path: Path, scene: dict[str, dict]) -> None:
    """Write *scene* to *path* (ENVIRONMENT.md) as a JSON code block.

    Preserves the human-readable header so that the LLM agent can still
    understand the file's purpose.
    """
    scene_json = json.dumps(scene, indent=2, ensure_ascii=False)
    content = (
        "# Environment Scene-Graph\n\n"
        "Auto-updated by HAL Watchdog after each action execution.\n"
        "Edit the JSON block below to set up or reset the test scene.\n\n"
        f"{_FENCE_OPEN}\n{scene_json}\n{_FENCE_CLOSE}\n"
    )
    path.write_text(content, encoding="utf-8")
