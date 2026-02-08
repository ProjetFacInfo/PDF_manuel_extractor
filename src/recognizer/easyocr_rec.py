import easyocr
from .base import BaseRecognizer
from src.utils.config import USE_GPU


class EasyOcrRecognizer(BaseRecognizer):
    """
    Stratégie de reconnaissance basée sur EasyOCR.
    EasyOCR est une bibliothèque OCR populaire qui utilise des réseaux de neurones profonds pour détecter et reconnaître le texte dans les images.
    Elle est efficace pour les textes de différentes tailles et orientations,
    ce qui en fait un choix solide pour la reconnaissance de texte dans des images complexes.
    """
    def __init__(self):
        print("Recognize : EasyOCR")
        self.reader = easyocr.Reader(['en'], gpu=USE_GPU)

    def recognize(self, crop_cv):
        res = self.reader.recognize(crop_cv, detail=0)
        return res[0] if res else ""