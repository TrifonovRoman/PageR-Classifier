import unittest
from pager import PageModel, PageModelUnit, PhisicalModel, WordsToOneBlock
from scr.base_classifier import BaseClassifier
class TestBaseClassifier(unittest.TestCase):
    def setUp(self):
        self.page = PageModel(page_units=[
            PageModelUnit(id="phisical_model",
                          sub_model=PhisicalModel(),
                          extractors=[],
                          converters={"words_model": WordsToOneBlock()})
        ])

        self.page.read_from_file('../../PageR-main/tests/files/blocks.json')

        self.extractor = BaseClassifier()

    def test_extract_changes_label_to_text(self):
        model = self.page.page_units[0].sub_model

        self.extractor.extract(model)

        for block in model.blocks:
            self.assertEqual(block.label, "text", f"Label was not changed to 'text' for block {block}")

    if __name__ == "__main__":
        unittest.main()