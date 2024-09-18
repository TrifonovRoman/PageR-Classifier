from pager import BaseExtractor, BaseSubModel
class BaseClassifier(BaseExtractor):
    def extract(self, model: BaseSubModel) -> None:
        for block in model.blocks:
            block.label = "text"