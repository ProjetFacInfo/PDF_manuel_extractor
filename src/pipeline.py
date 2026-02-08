import os
import cv2
import json
from src.utils.visualizer import VisualUtils
from src.utils.corrector import TextCorrector
from src.detector.base import BaseDetector
from src.recognizer.base import BaseRecognizer

class CadPipeline:
    """
        Orchestre le processus complet d'extraction en combinant dynamiquement un Détecteur et un Reconnaisseur.
        Elle gère le cycle de vie complet : pré-traitement (upscaling), détection, découpage (cropping), reconnaissance, correction des erreurs et export des résultats.
    """
    def __init__(self, detector: BaseDetector, recognizer: BaseRecognizer, scale=3.0):
        self.detector = detector
        self.recognizer = recognizer
        self.scale = scale

    def process(self, image_path):
        if not os.path.exists(image_path):
            print(f"❌ Fichier introuvable : {image_path}")
            return

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(image_path)
        big_img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        boxes = self.detector.detect(big_img)
        extracted_data = []
        annotated_img = img.copy()

        for (bx_min, bx_max, by_min, by_max) in boxes:
            pad = 12
            h, w, _ = big_img.shape
            y1, y2 = max(0, by_min - pad), min(h, by_max + pad)
            x1, x2 = max(0, bx_min - pad), min(w, bx_max + pad)

            crop = big_img[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10: continue

            try:
                raw_text = self.recognizer.recognize(crop)
                final_text = TextCorrector.fix_common_errors(raw_text)

                if len(final_text) > 0 and final_text != ".":
                    orig_box = [int(c / self.scale) for c in [x1, y1, x2, y2]]

                    extracted_data.append({
                        "text": final_text, "raw_text": raw_text, "box": orig_box
                    })

                    cv2.rectangle(annotated_img, (orig_box[0], orig_box[1]), (orig_box[2], orig_box[3]), (0, 255, 0), 1)
                    color = (255, 0, 0) if len(final_text) == 1 and final_text in "ABC" else (0, 0, 255)
                    annotated_img = VisualUtils.draw_text_unicode(annotated_img, final_text, (orig_box[0], orig_box[1] - 18), color)

            except Exception: pass

        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        suffix = f"_{type(self.detector).__name__}_{type(self.recognizer).__name__}"
        out_img_path = os.path.join(output_dir, f"out{suffix}_{base_name}.png")
        out_json_path = os.path.join(output_dir, f"out{suffix}_{base_name}.json")

        cv2.imwrite(out_img_path, annotated_img)
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)

        return out_json_path
