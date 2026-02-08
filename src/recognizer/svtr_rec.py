from rapidocr_onnxruntime import RapidOCR
from .base import BaseRecognizer

class SvtrRecognizer(BaseRecognizer):
    """
        Utilise l'architecture SVTR (Single Visual Transformer) via le moteur léger RapidOCR (ONNX).
        C'est le modèle le plus rapide du projet, offrant un excellent compromis vitesse/précision pour le traitement de masse.
    """
    def __init__(self):
        print("Recognizer : SVTR")
        self.engine = RapidOCR()

    def recognize(self, crop_cv):
        result, _ = self.engine(crop_cv, use_det=False, use_rec=True, use_cls=False)
        return result[0][0] if result else ""