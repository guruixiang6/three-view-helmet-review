import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import numpy as np

import evaluate
from helmet_pipeline import AnJianDetection
from vlm_review import VisionReviewer


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((16, 20, 3), dtype=np.uint8)
        self.reviewer = VisionReviewer("http://127.0.0.1:8000/v1", "test-vision-model")
        self.reviewer.opener = Mock()

    def reply(self, content):
        raw = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.reviewer.opener.open.return_value = io.BytesIO(raw)

    def test_request_and_reply(self):
        self.reply('```json\n{"decision":"violation"}\n```')
        self.assertEqual(self.reviewer.review(self.image), ("violation", "resolved"))
        request = self.reviewer.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertIsNone(request.get_header("Authorization"))
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "test-vision-model")
        self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_unusable_responses_abstain(self):
        for content, status in (("not JSON", "invalid_response"), ('{"decision":"other"}', "invalid_response"),
                                ('{"decision":"uncertain"}', "abstained")):
            with self.subTest(content=content):
                self.reply(content)
                self.assertEqual(self.reviewer.review(self.image), ("uncertain", status))
        for exc, status in ((TimeoutError(), "request_error"),
                            (HTTPError("", 429, "", {}, io.BytesIO()), "http_429")):
            self.reviewer.opener.open.side_effect = exc
            self.assertEqual(self.reviewer.review(self.image), ("uncertain", status))

    def test_evaluation_calls_vlm_only_for_uncertain_evaluable_images(self):
        detector = AnJianDetection.__new__(AnJianDetection)
        detector.debug_logging = False
        detector._run_group_detection = Mock(return_value={"coords": [], "categories": [], "scores": []})
        stress = self.image.copy()
        detector.build_stress_view = Mock(return_value=stress)
        detector._save_auxiliary_views = Mock(return_value={})
        detector._compute_risk = Mock(side_effect=[
            {"needs_review": True, "route_reason": "ambiguous"},
            {"needs_review": False, "suggested_decision": "no_violation"},
        ])
        reviewer = Mock()
        reviewer.review.return_value = "violation", "resolved"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "test/images").mkdir(parents=True)
            (root / "test/labels").mkdir()
            model = root / "best.pt"
            model.touch()
            for name, label in (("a", 1), ("b", 0), ("c", 2)):
                (root / f"test/images/{name}.jpg").touch()
                (root / f"test/labels/{name}.txt").write_text(f"{label} .5 .5 .1 .1\n", encoding="ascii")
            out = root / "report"
            args = ["evaluate.py", "--model", str(model), "--method", "three-view", "--data", str(root),
                    "--out", str(out), "--vlm-review"]
            with patch("sys.argv", args), patch("vlm_review.VisionReviewer.from_environment", return_value=reviewer), \
                 patch("helmet_pipeline.AnJianDetection", return_value=detector), \
                 patch("helmet_pipeline.load_image", return_value=self.image), redirect_stdout(io.StringIO()):
                evaluate.main()
            report = json.loads((out / "metrics.json").read_text())
            with (out / "per_image.csv").open(newline="") as file:
                first = next(csv.DictReader(file))
        reviewer.review.assert_called_once()
        self.assertIs(reviewer.review.call_args.args[0], stress)
        self.assertEqual(report["reviewed_count"], 1)
        self.assertEqual(report["before_review"]["uncertain"], 1)
        self.assertEqual(report["after_review"]["tp"], 1)
        self.assertEqual(report["after_review"]["tn"], 1)
        self.assertEqual(report["after_review"]["excluded"], 1)
        self.assertEqual((first["prediction"], first["final_prediction"], first["review_view"]),
                         ("uncertain", "violation", "stress"))


if __name__ == "__main__":
    unittest.main()
