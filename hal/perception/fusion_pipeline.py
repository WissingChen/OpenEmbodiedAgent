"""Fusion pipeline skeleton for building scene graph nodes."""

from __future__ import annotations

from datetime import datetime


class FusionPipeline:
    """Fuses detections into a structured scene graph."""

    def process(self, detections: list[dict], geometry: dict | None = None) -> dict:
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        nodes = []
        for idx, detection in enumerate(detections):
            nodes.append(
                {
                    "id": f"det_{idx}",
                    "class": detection.get("label", "unknown"),
                    "confidence": detection.get("confidence", 0.0),
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "size": {"x": 0.5, "y": 0.5, "z": 0.5},
                    "frame": "map",
                    "track_id": f"track_{idx}",
                    "last_seen_at": timestamp,
                }
            )
        return {"nodes": nodes, "edges": []}
