"""Side-loaded perception daemon with fake-data friendly inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hal.perception.environment_writer import EnvironmentWriter, PerceptionEventSpec
from hal.perception.fusion_pipeline import FusionPipeline
from hal.perception.geometry_pipeline import GeometryPipeline
from hal.perception.segmentation_pipeline import SegmentationPipeline

if TYPE_CHECKING:
    from PhyAgentOS.bus.queue import MessageBus


class PerceptionService:
    """Coordinates geometry, segmentation, fusion, and environment writes.

    Parameters
    ----------
    workspace:
        Directory containing ``ENVIRONMENT.md``.
    bus:
        Optional ``MessageBus``.  When provided, noteworthy perception events
        (e.g. new object detected, target lost) are published to
        ``bus.perception`` so the AgentLoop can react proactively.
    """

    def __init__(self, workspace: Path, bus: "MessageBus | None" = None):
        self.workspace = workspace
        self.geometry    = GeometryPipeline()
        self.segmentation = SegmentationPipeline()
        self.fusion      = FusionPipeline()
        self.writer      = EnvironmentWriter(workspace, bus=bus)

    def tick(
        self,
        *,
        robot_id: str,
        image: Any = None,
        pointcloud: Any = None,
        odom: dict | None = None,
        nav_state: dict | None = None,
        events: list[PerceptionEventSpec] | None = None,
    ) -> dict:
        """Run one perception cycle and write results to ENVIRONMENT.md.

        Parameters
        ----------
        robot_id:
            ID of the robot being perceived.
        image, pointcloud, odom, nav_state:
            Raw sensor inputs (may be ``None`` for unused modalities).
        events:
            Optional list of :class:`PerceptionEventSpec` to publish after
            the environment write.  Callers should populate this when they
            detect a noteworthy change (e.g. a new object class appeared in
            the scene graph).
        """
        geometry    = self.geometry.process(pointcloud=pointcloud, odom=odom)
        detections  = self.segmentation.process(image=image)
        scene_graph = self.fusion.process(detections=detections, geometry=geometry)
        return self.writer.write(
            robot_id=robot_id,
            robot_pose=odom,
            nav_state=nav_state,
            scene_graph=scene_graph,
            map_data=geometry.get("map"),
            tf_data=geometry.get("tf"),
            events=events,
        )
