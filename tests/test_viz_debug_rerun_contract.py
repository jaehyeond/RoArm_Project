from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from roarm_rl.rerun_contract import validate_rerun_artifact
from roarm_rl.viz_debug import log_rerun


class RerunContractTest(unittest.TestCase):
    def _frame(self, name: str = "target") -> dict[str, object]:
        return {"name": name, "position": [0.10, 0.20, 0.30]}

    def test_frames_only_is_finalized_and_footer_verified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roarm_rerun_test_") as tmp:
            path = Path(tmp) / "frames.rrd"
            status = log_rerun(path, frames=[self._frame()])
            self.assertTrue(status["ok"], status)
            self.assertTrue(status["sink_attached_before_logging"])
            self.assertTrue(status["sink_finalized"])
            self.assertTrue(status["archive_validation"]["footer_manifest_present"])
            self.assertTrue(status["archive_validation"]["pass"])
            self.assertFalse(status["visual_inspection_complete"])
            self.assertFalse(status["completion_contract_pass"])
            frame_schema = validate_rerun_artifact(
                path,
                expected_entity_components={
                    "frames/target": [
                        "Transform3D:child_frame",
                        "Transform3D:parent_frame",
                        "Transform3D:quaternion",
                        "Transform3D:translation",
                    ],
                    "frames/target/origin": [
                        "CoordinateFrame:frame",
                        "Points3D:positions",
                    ],
                },
            )
            self.assertTrue(frame_schema["pass"], frame_schema)

    def test_two_sequential_recordings_finalize_independently(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roarm_rerun_test_") as tmp:
            paths = [Path(tmp) / "first.rrd", Path(tmp) / "second.rrd"]
            statuses = [
                log_rerun(path, frames=[self._frame(path.stem)])
                for path in paths
            ]
            self.assertTrue(all(status["ok"] for status in statuses), statuses)
            for path in paths:
                validation = validate_rerun_artifact(path)
                self.assertTrue(validation["pass"], validation)

    def test_invalid_mesh_is_rejected_before_sink_creation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roarm_rerun_test_") as tmp:
            path = Path(tmp) / "invalid.rrd"
            status = log_rerun(
                path,
                meshes=[
                    {
                        "entity_path": "geometry/invalid",
                        "coordinate_frame": "world",
                        "vertices_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "triangles": [[0, 1, 3]],
                    }
                ],
            )
            self.assertFalse(status["ok"])
            self.assertIn("out of range", status["error"])
            self.assertFalse(path.exists())

    def test_undeclared_spatial_coordinate_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roarm_rerun_test_") as tmp:
            path = Path(tmp) / "undeclared_frame.rrd"
            status = log_rerun(
                path,
                points=[
                    {
                        "entity_path": "contacts/points",
                        "coordinate_frame": "tool_surface",
                        "positions_m": [[0.0, 0.0, 0.0]],
                    }
                ],
            )
            self.assertFalse(status["ok"])
            self.assertIn("spatial coordinate frames were not declared", status["error"])
            self.assertFalse(path.exists())

    def test_mesh_scalar_event_and_footer_negative_control(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roarm_rerun_test_") as tmp:
            path = Path(tmp) / "collision.rrd"
            status = log_rerun(
                path,
                blueprint_mode="collision_gate",
                coordinate_frames=[
                    {
                        "frame": "link5_body_local",
                        "parent_frame": "tf#/",
                        "entity_path": "coordinate_frames/link5_body_local",
                    }
                ],
                meshes=[
                    {
                        "entity_path": "cook/source/link5/parts/part_000",
                        "coordinate_frame": "link5_body_local",
                        "vertices_m": [
                            [0.0, 0.0, 0.0],
                            [0.01, 0.0, 0.0],
                            [0.0, 0.01, 0.0],
                            [0.0, 0.0, 0.01],
                        ],
                        "triangles": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                    }
                ],
                scalar_trace=[
                    {
                        "entity_path": "metrics/link5/part_000/coordinate_delta_m",
                        "value": 0.0,
                        "sequence": {"event_idx": 0, "part_idx": 0},
                    }
                ],
                events=[
                    {
                        "entity_path": "events/cook",
                        "text": "RESULT_VALID",
                        "level": "INFO",
                        "sequence": {"event_idx": 0, "part_idx": 0},
                    }
                ],
            )
            self.assertTrue(status["ok"], status)
            self.assertEqual(status["mesh_count"], 1)
            self.assertEqual(status["archive_validation"]["timeline_contract"]["observed"], [
                "blueprint",
                "event_idx",
                "log_time",
                "part_idx",
            ])
            exact = validate_rerun_artifact(
                path,
                exact_entity_paths=[
                    "coordinate_frames/link5_body_local",
                    "cook/source/link5/parts/part_000",
                    "metadata/meshes/cook__source__link5__parts__part_000",
                    "metadata/run",
                    "metrics/link5/part_000/coordinate_delta_m",
                    "events/cook",
                ],
                exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
                expected_entity_components={
                    "coordinate_frames/link5_body_local": [
                        "Transform3D:child_frame",
                        "Transform3D:parent_frame",
                        "Transform3D:quaternion",
                        "Transform3D:translation",
                    ],
                    "cook/source/link5/parts/part_000": [
                        "CoordinateFrame:frame",
                        "Mesh3D:albedo_factor",
                        "Mesh3D:triangle_indices",
                        "Mesh3D:vertex_positions",
                    ],
                    "metrics/link5/part_000/coordinate_delta_m": ["Scalars:scalars"],
                    "events/cook": ["TextLog:level", "TextLog:text"],
                },
            )
            self.assertTrue(exact["pass"], exact)

            unexpected = validate_rerun_artifact(
                path,
                exact_entity_paths=["metadata/run"],
                exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
            )
            self.assertFalse(unexpected["pass"], unexpected)
            self.assertIn(
                "/cook/source/link5/parts/part_000",
                unexpected["entity_path_contract"]["unexpected_non_system"],
            )

            truncated = Path(tmp) / "collision_truncated.rrd"
            truncated.write_bytes(path.read_bytes()[:-256])
            negative = validate_rerun_artifact(truncated)
            self.assertFalse(negative["pass"], negative)
            self.assertFalse(negative["footer_manifest_present"])
            self.assertIn("RRD footer verification failed", negative["errors"])

    def test_urdf_prefix_is_not_escaped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="roarm_rerun_test_") as tmp:
            root = Path(tmp)
            urdf = root / "prefix_probe.urdf"
            urdf.write_text(
                """<?xml version="1.0"?>
<robot name="prefix_probe">
  <link name="base_link">
    <visual><geometry><box size="0.01 0.01 0.01"/></geometry></visual>
  </link>
</robot>
""",
                encoding="utf-8",
            )
            path = root / "urdf.rrd"
            status = log_rerun(path, urdf_path=urdf, frames=[self._frame()])
            self.assertTrue(status["ok"], status)
            stats = status["archive_validation"]["stats"]["stdout"]
            self.assertIn("/actual_robot/prefix_probe", stats)
            self.assertNotIn("/\\/actual_robot", stats)


if __name__ == "__main__":
    unittest.main()
