"""Missing OS telemetry must not falsify measurements or block the science."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from criteo_release.runtime import Measurement


class MeasurementTests(unittest.TestCase):
    def test_unavailable_process_memory_keeps_timing_and_null_rss(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "criteo_release.runtime.psutil.Process", side_effect=OSError("no process filesystem")
        ):
            out = Path(tmp) / "runtime.json"
            measure = Measurement(out)
            with measure.stage("calculation"):
                self.assertEqual(sum(range(10)), 45)
            result = json.loads(out.read_text())
            self.assertIsNone(result["observed_process_peak_rss_bytes"])
            self.assertIsNone(result["stages"][0]["observed_peak_rss_bytes"])
            self.assertEqual(result["stages"][0]["rss_sample_count"], 0)
            self.assertGreaterEqual(result["stages"][0]["wall_seconds"], 0)
            self.assertIn("no process filesystem", result["rss_measurement_error"])

    def test_sampling_failure_does_not_hide_calculation_failure(self):
        process = Mock()
        process.memory_info.side_effect = PermissionError("RSS denied")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "criteo_release.runtime.psutil.Process", return_value=process
        ):
            measure = Measurement(Path(tmp) / "runtime.json")
            with self.assertRaisesRegex(ValueError, "scientific failure"):
                with measure.stage("failed calculation"):
                    raise ValueError("scientific failure")
            self.assertIn("RSS denied", measure.memory_error)

    def test_available_rss_is_measured(self):
        process = Mock()
        process.memory_info.return_value.rss = 2048
        with tempfile.TemporaryDirectory() as tmp, patch(
            "criteo_release.runtime.psutil.Process", return_value=process
        ):
            out = Path(tmp) / "runtime.json"
            with Measurement(out).stage("calculation"):
                pass
            result = json.loads(out.read_text())
            self.assertEqual(result["observed_process_peak_rss_bytes"], 2048)
            self.assertEqual(result["rss_measurement_status"], "sampled")
