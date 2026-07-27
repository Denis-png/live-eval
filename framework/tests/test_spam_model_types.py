import unittest
from unittest.mock import patch

from framework.tasks.spam.task import SpamTask


class SpamModelTypeDispatchTests(unittest.TestCase):
    """get_model must dispatch the new bert types and merge the per-type
    params (label_map, max_length) from configs/spam/spam.json."""

    def _get_merged(self, name, model_type):
        captured = {}

        def fake_load(self, model_config):
            captured.update(model_config)

        with patch("framework.models.spam.bert_tiny.BertTinySpamModel.load_model", fake_load), \
             patch("framework.models.spam.roberta.RobertaSpamModel.load_model", fake_load):
            SpamTask().get_model({"name": name, "type": model_type})
        return captured

    def test_bert_tiny_type_maps_generic_labels(self):
        merged = self._get_merged("mrm8488/bert-tiny-finetuned-sms-spam-detection", "bert_tiny")
        self.assertEqual(merged["label_map"], {"LABEL_0": "HAM", "LABEL_1": "SPAM"})
        self.assertEqual(merged["max_length"], 128)

    def test_bert_type_passes_native_labels_through(self):
        merged = self._get_merged("wesleyacheng/sms-spam-classification-with-bert", "bert")
        self.assertEqual(merged["label_map"], {"HAM": "HAM", "SPAM": "SPAM"})

    def test_roberta_type_still_supported(self):
        merged = self._get_merged("mshenoda/roberta-spam", "roberta")
        self.assertEqual(merged["label_map"], {"LABEL_0": "HAM", "LABEL_1": "SPAM"})

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            SpamTask().get_model({"name": "x", "type": "nope"})


if __name__ == "__main__":
    unittest.main()
