import unittest

from framework.profiling.spam_profiler import detect_signals, analyze_spam_signals


class DetectSignalsTests(unittest.TestCase):
    def test_each_signal_fires_on_positive_text(self):
        self.assertIn("phishing_link", detect_signals("click http://evil.com now"))
        self.assertIn("money_promise", detect_signals("win $5000 today"))
        self.assertIn("excessive_caps", detect_signals("FREE STUFF here"))
        self.assertIn("urgency", detect_signals("act now!"))
        self.assertIn("spam_keywords", detect_signals("you are a winner"))

    def test_clean_text_has_no_signals(self):
        self.assertEqual(detect_signals("see you at the meeting tomorrow"), set())

    def test_analyze_spam_signals_unchanged_output_keys(self):
        rows = [{"text": "win $5000 http://x.com NOW!", "label": "SPAM"},
                {"text": "lunch tomorrow?", "label": "HAM"}]
        out = analyze_spam_signals(rows)
        self.assertEqual(
            set(out["SPAM"].keys()),
            {"url_rate", "currency_rate", "caps_rate", "exclaim_rate", "keyword_rate"},
        )
        self.assertEqual(out["SPAM"]["url_rate"], 1.0)
