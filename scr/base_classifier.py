from pager import BaseExtractor, BaseSubModel
import tensorflow as tf
class BaseClassifier(BaseExtractor):
    def __init__(self, save_path: str):
        super().__init__()
        self.mlp_loaded = tf.saved_model.load(save_path)
    def extract(self, model: BaseSubModel) -> None:
        for block, graph in zip(model.blocks,model.graphs):
            block.label = self._classification(graph)

    def _classification(self, graph):
        result = self.mlp_loaded(graph)
        return "text" if result > 0.5 else "header"