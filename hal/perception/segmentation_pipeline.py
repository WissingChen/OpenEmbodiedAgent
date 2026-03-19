"""Segmentation pipeline skeleton."""

from __future__ import annotations

from typing import Any


class SegmentationPipeline:
    """Produces semantic detections from RGB frames."""

    def process(self, image: Any = None) -> list[dict]:
        return [] if image is None else [{"label": "unknown", "confidence": 0.0}]
