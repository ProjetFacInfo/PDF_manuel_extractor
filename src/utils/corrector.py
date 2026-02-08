import re

class TextCorrector:
    """Logique métier pour nettoyer les erreurs OCR spécifiques à la CAO"""
    @staticmethod
    def fix_common_errors(text):
        text = text.strip().strip('-')
        if not text: return ""

        # Degrés (4X0" -> 4X0°), UTF-8
        text = re.sub(r'(\d+)["\']', r'\1°', text)
        text = re.sub(r'(\d+)0"', r'\1°', text)
        text = text.replace("º", "°")

        return text