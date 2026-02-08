import abc

class BaseDetector(abc.ABC):
    """
        Interface abstraite définissant le contrat obligatoire pour tous les modèles de détection de texte.
        Toute classe héritant de celle-ci doit implémenter la méthode detect qui prend une image et retourne une liste de coordonnées de boîtes englobantes.
    """
    @abc.abstractmethod
    def detect(self, img_cv):
        """
        Entrée : Image OpenCV (BGR)
        Sortie : Liste de tuples [(x_min, x_max, y_min, y_max), ...]
        """
        pass