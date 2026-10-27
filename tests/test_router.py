
import unittest

from app.orchestrator.router import AdaptiveRouter


class TestAdaptiveRouter(unittest.TestCase):
    def setUp(self):
        self.router = AdaptiveRouter()

    def test_cleaning_request(self):
        self.assertEqual(
            self.router.route("Clean my dataset"),
            "DataAgent"
        )

    def test_eda_request(self):
        self.assertEqual(
            self.router.route("Find correlations in my data"),
            "EDAAgent"
        )

    def test_prediction_request(self):
        self.assertEqual(
            self.router.route("Predict customer churn"),
            "MLAgent"
        )

    def test_report_request(self):
        self.assertEqual(
            self.router.route("Generate a report"),
            "ReportAgent"
        )


if __name__ == "__main__":
    unittest.main()
