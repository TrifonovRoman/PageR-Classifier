import unittest
import tempfile
import os
import json
import shutil

from pager.benchmark.seg_detection.seg_detection import SegDetectionBenchmark, LABELS
if "title" not in LABELS: #без этого будет ошибка
    LABELS["title"] = 1


class PageModel:
    def __init__(self, predictions_map):
        self.predictions_map = predictions_map
        self.current_file = None

    def read_from_file(self, path):
        self.current_file = os.path.basename(path)

    def extract(self):
        pass

    def to_dict(self):
        return {"blocks": self.predictions_map.get(self.current_file, [])}

class TestSegDetectionBenchmark(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        for img_name in ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"]:
            with open(os.path.join(self.test_dir, img_name), "wb") as f:
                f.write(b"")

        self.dataset = {
            "categories": [
                {"id": 0, "name": "text"},
                {"id": 1, "name": "title"},
                {"id": 2, "name": "list"},
                {"id": 3, "name": "table"},
                {"id": 4, "name": "figure"}
            ],
            "images": [
                {"id": 0, "file_name": "img1.png"},
                {"id": 1, "file_name": "img2.png"},
                {"id": 2, "file_name": "img3.png"},
                {"id": 3, "file_name": "img4.png"},
                {"id": 4, "file_name": "img5.png"}
            ],
            "annotations": [
                {"image_id": 0, "category_id": 0, "bbox": [10, 10, 100, 100]},
                {"image_id": 0, "category_id": 1, "bbox": [150, 150, 80, 80]},
                {"image_id": 1, "category_id": 0, "bbox": [20, 20, 50, 50]},
                {"image_id": 1, "category_id": 1, "bbox": [100, 100, 60, 60]},
                {"image_id": 2, "category_id": 2, "bbox": [30, 30, 70, 70]},
                {"image_id": 3, "category_id": 3, "bbox": [40, 40, 120, 120]},
                {"image_id": 4, "category_id": 4, "bbox": [50, 50, 90, 90]},
                {"image_id": 4, "category_id": 0, "bbox": [200, 200, 50, 50]}
            ]
        }
        self.json_path = os.path.join(self.test_dir, "dataset.json")
        with open(self.json_path, "w") as f:
            json.dump(self.dataset, f)

        self.predictions_map = {
            "img1.png": [
                {"label": "text", "x_top_left": 10, "y_top_left": 10, "x_bottom_right": 110, "y_bottom_right": 110},
                {"label": "title", "x_top_left": 150, "y_top_left": 150, "x_bottom_right": 225, "y_bottom_right": 225}
            ],
            "img2.png": [
                # Ложные предсказания – блоки
                {"label": "text", "x_top_left": 1000, "y_top_left": 1000, "x_bottom_right": 1010, "y_bottom_right": 1010},
                {"label": "title", "x_top_left": 0, "y_top_left": 0, "x_bottom_right": 10, "y_bottom_right": 10}
            ],
            "img3.png": [
                # частичное пересечение для списка
                {"label": "list", "x_top_left": 35, "y_top_left": 35, "x_bottom_right": 95, "y_bottom_right": 95}
            ],
            "img4.png": [
                # без предсказаний
            ],
            "img5.png": [
                # figure – точное совпадение, text – с неполное
                {"label": "figure", "x_top_left": 50, "y_top_left": 50, "x_bottom_right": 140, "y_bottom_right": 140},
                {"label": "text", "x_top_left": 210, "y_top_left": 210, "x_bottom_right": 260, "y_bottom_right": 260}
            ]
        }
        self.dummy_page_model = PageModel(self.predictions_map)

        self.logged_messages = []

        self.benchmark = SegDetectionBenchmark(
            path_dataset=self.test_dir,
            page_model=self.dummy_page_model,
            path_json=self.json_path,

        )
        self.benchmark.loger = lambda msg: self.logged_messages.append(msg)

    def tearDown(self):
        shutil.rmtree(self.test_dir)


    def test_calculate_iou(self):
        # Полное совпадение
        iou = self.benchmark.calculate_iou([10, 10, 100, 100], [10, 10, 100, 100])
        self.assertAlmostEqual(iou, 1.0, places=4)

        # Отсутствие пересечения
        iou = self.benchmark.calculate_iou([10, 10, 100, 100], [200, 200, 50, 50])
        self.assertEqual(iou, 0.0)

        # Частичное пересечение:
        iou = self.benchmark.calculate_iou([10, 10, 100, 100], [50, 50, 100, 100])
        expected_iou = 3600 / 16400
        self.assertAlmostEqual(iou, expected_iou, places=4)

if __name__ == '__main__':
    unittest.main()
