from pager import PhisicalModel
from typing import List, Tuple
import numpy as np
class GNNPhisicalModel(PhisicalModel):
    def __init__(self) -> None:
        super().__init__()
        self.graphs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []