import os
import time

from src.utils.config import DEVICE, USE_GPU

# pip install tabulate pandas
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from src.pipeline import CadPipeline
from src.detector.easyocr_det import EasyOcrDetector
from src.detector.dbnet_det import DbNetDetector
from src.recognizer.easyocr_rec import EasyOcrRecognizer
from src.recognizer.trocr_rec import TrOcrRecognizer
from src.recognizer.parseq_rec import ParseqRecognizer
from src.recognizer.svtr_rec import SvtrRecognizer

if __name__ == "__main__":

    INPUT_IMAGE = os.path.join("data", "SOT-23.png")

    if not os.path.exists(INPUT_IMAGE):
        print(f"ERREUR : {INPUT_IMAGE} introuvable.")
        exit(1)

    print(f"Chargement des modèles sur {DEVICE}...")

    # Instanciation
    detectors = [
        EasyOcrDetector(),
        DbNetDetector()
    ]

    recognizers = [
        EasyOcrRecognizer(),
        TrOcrRecognizer(),
        ParseqRecognizer(),
        SvtrRecognizer()
    ]

    results = []
    SCALE = 4.0

    print(f"\nDÉBUT DU BENCHMARK SUR {INPUT_IMAGE} ({DEVICE}) 🔥\n")

    for detector in detectors:
        for recognizer in recognizers:

            det_name = type(detector).__name__
            rec_name = type(recognizer).__name__

            print(f"Processing : {det_name} + {rec_name}...", end=" ", flush=True)

            start_time = time.time()

            try:
                pipeline = CadPipeline(detector, recognizer, scale=SCALE)
                json_path = pipeline.process(INPUT_IMAGE)

                duration = time.time() - start_time

                item_count = 0
                if json_path and os.path.exists(json_path):
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        item_count = len(data)

                print(f"({duration:.2f}s) -> {item_count} items")

                results.append({
                    "Detector": det_name.replace("Detector", ""),
                    "Recognizer": rec_name.replace("Recognizer", ""),
                    "Time (s)": round(duration, 2),
                    "Items": item_count,
                    "Device": str(DEVICE)
                })

            except Exception as e:
                print(f"\nErreur : {e}")
                results.append({
                    "Detector": det_name, "Recognizer": rec_name, "Time (s)": "FAIL", "Items": 0
                })

    print("\n" + "="*60)
    print(f"RÉSULTATS DU BENCHMARK ({DEVICE})")
    print("="*60)

    if tabulate:
        print(tabulate(results, headers="keys", tablefmt="github"))
    else:
        for r in results:
            print(r)
