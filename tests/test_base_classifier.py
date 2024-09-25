import unittest
from pager import PhisicalModel
from scr.base_classifier import BaseClassifier
class TestBaseClassifier(unittest.TestCase):
    def setUp(self):
        self.phisical_model = PhisicalModel()


        self.phisical_model.read_from_file('tests/files/blocks.json')

        self.extractor = BaseClassifier()

    def test_extract_changes_label_to_text(self):
        model = self.phisical_model

        self.extractor.extract(model)

        for block in model.blocks:
            self.assertEqual(block.label, "text", f"Label was not changed to 'text' for block {block}")

    if __name__ == "__main__":
        unittest.main()