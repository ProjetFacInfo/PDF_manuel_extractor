import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from .base import BaseRecognizer
from src.utils.config import DEVICE

class TrOcrRecognizer(BaseRecognizer):
    """
        Utilise le modèle Transformer TrOCR de Microsoft pour une lecture optique de haute précision.
        Bien que plus lent, il excelle dans le déchiffrage de textes flous ou bruités en utilisant une approche générative pixel-par-pixel.
    """
    def __init__(self):
        print("Stratégie Lecture : TrOCR (Microsoft)")
        model_name = "microsoft/trocr-base-printed"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)

    def recognize(self, crop_cv):
        img_pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))
        pixel_values = self.processor(images=img_pil, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]