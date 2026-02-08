import abc

class BaseRecognizer(abc.ABC):
    """
        Interface abstraite définissant le contrat pour tous les modèles de reconnaissance de caractères.
        Elle impose l'implémentation de la méthode `recognize` qui reçoit une image rognée (crop) et doit retourner la chaîne de caractères brute lue.
    """
    @abc.abstractmethod
    def recognize(self, crop_cv):
        """
        Entrée : Crop Image OpenCV (BGR)
        Sortie : String (Texte brut)
        """
        pass