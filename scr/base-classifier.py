import sys
import os
sys.path.append(os.path.abspath('../PageR-main'))
from pager.page_model.sub_models.base_sub_model import BaseExtractor, BaseSubModel
class BaseClassifier(BaseExtractor):
    def extract(self, model: BaseSubModel) -> None:
        for block in model.blocks:
            block.set_label("text")