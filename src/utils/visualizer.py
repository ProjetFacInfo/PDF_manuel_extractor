import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class VisualUtils:
    """
    Gère l'affichage avancé du texte sur les images en utilisant Pillow pour contourner les limitations d'OpenCV.
    Cette classe permet d'incruster correctement les caractères Unicode (comme le symbole degré °) et d'ajouter un fond contrasté.
    """

    @staticmethod
    def draw_text_unicode(img_cv, text, position, color=(0, 255, 0), font_size=16):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        try:
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
        except AttributeError:
            pass

        # 4. Dessin du texte
        # Note: color[::-1] n'est pas nécessaire ici car on dessine sur du RGB (PIL) avec une couleur RGB.
        # Si ta couleur d'entrée est (0, 255, 0) pour Vert, c'est bon.
        # Si ta couleur d'entrée est BGR (OpenCV standard), il faut inverser.
        # Dans ton main, tu passes (0, 255, 0) ou (255, 0, 0), supposons que c'est du RGB désiré.
        draw.text(position, text, font=font, fill=color[::-1])

        # 5. RETOUR CORRIGÉ : On renvoie l'image PIL convertie en BGR pour OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
