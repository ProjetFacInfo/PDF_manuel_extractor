import easyocr
from .base import BaseDetector
from src.utils.config import USE_GPU # Import de la config

class EasyOcrDetector(BaseDetector):
    """
        Implémentation du détecteur utilisant l'algorithme CRAFT via la librairie EasyOCR.
        Ce détecteur est particulièrement efficace pour segmenter les petits blocs de texte isolés et les caractères techniques dispersés sur un plan.
    """
    def __init__(self):
        print(f"Detector : EasyOCR (GPU={USE_GPU})")
        self.reader = easyocr.Reader(['en'], gpu=USE_GPU, quantize=False)
        self.params = {'text_threshold': 0.4, 'link_threshold': 0.2, 'low_text': 0.35, 'slope_ths': 0.3}

    def detect(self, img_cv):
        horizontal, free = self.reader.detect(img_cv, **self.params)
        boxes = []
        for box in horizontal[0]: boxes.append(tuple(map(int, box)))
        for box in free[0]:
            x_c, y_c = [p[0] for p in box], [p[1] for p in box]
            boxes.append((int(min(x_c)), int(max(x_c)), int(min(y_c)), int(max(y_c))))
        return boxes
