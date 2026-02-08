import numpy as np
from rapidocr_onnxruntime import RapidOCR
from .base import BaseDetector
from src.utils.config import USE_GPU

class DbNetDetector(BaseDetector):
    """
        Implémentation du détecteur basée sur l'architecture DBNet++ exécutée via le moteur rapide RapidOCR (ONNX).
        Offre une détection extrêmement véloce et robuste, optimisée pour séparer les lignes de texte denses ou très rapprochées.
    """
    def __init__(self):
        print(f"Detector : DBNet++ (GPU={USE_GPU})")
        self.engine = RapidOCR(det_use_cuda=USE_GPU)

    def detect(self, img_cv):
        result, _ = self.engine(img_cv, use_det=True, use_rec=False, use_cls=False)
        boxes = []
        if result:
            for item in result:
                poly = np.array(item).astype(np.int32)
                boxes.append((np.min(poly[:, 0]), np.max(poly[:, 0]), np.min(poly[:, 1]), np.max(poly[:, 1])))
        return boxes
