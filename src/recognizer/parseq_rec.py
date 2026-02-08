import cv2
import torch
import nltk
from PIL import Image
from torchvision import transforms as T
from .base import BaseRecognizer
from src.utils.config import DEVICE


class ParseqRecognizer(BaseRecognizer):
    """
        Exploite le modèle SOTA PARSeq (Permuted Autoregressive Sequence) via PyTorch pour une reconnaissance robuste.
        Il est capable de lire efficacement du texte orienté ou déformé grâce à son mécanisme d'attention contextuelle bidirectionnelle.
    """
    def __init__(self):
        print("Recognizer : PARSeq")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        self.model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).eval().to(DEVICE)
        self.preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

    def recognize(self, crop_cv):
        img_pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred = logits.softmax(-1)
            label, _ = self.model.tokenizer.decode(pred)
        return label[0]