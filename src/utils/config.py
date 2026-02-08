import os
import torch
import sys


def setup_environment():
    """
    Configure automatiquement l'environnement pour AMD, NVIDIA ou CPU.
    Si on détecte du AMD, on applique tes patchs spécifiques
    Pour NVIDIA ou CPU, on ne touche à rien
    """
    is_amd_rocm = os.path.exists('/dev/kfd')

    if is_amd_rocm:
        print("Matériel AMD ROCm détecté : Application des patchs RDNA2...")
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        os.environ["MIOPEN_DEBUG_COMGR_HIP_PCH_ENFORCE"] = "0"
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"
    else:
        pass


setup_environment()


def get_device():
    """
        Detecte automatiquement le device disponible : ROCm pour AMD, CUDA pour NVIDIA, ou CPU par défaut.
    """
    if torch.cuda.is_available():   # rocm | nvidia
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_device()
USE_GPU = (DEVICE.type != 'cpu')

print(f"Configuration Matérielle : {DEVICE} (GPU Activé: {USE_GPU})")