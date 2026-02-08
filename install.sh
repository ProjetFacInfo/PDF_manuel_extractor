#!/bin/bash

set -e

echo "--- Demarrage de l'installation automatique ---"

if [ ! -d ".venv" ]; then
    echo "Creation de l'environnement virtuel (.venv)..."
    python3 -m venv .venv
else
    echo "Environnement virtuel deja present."
fi

echo "Activation de l'environnement..."
source .venv/bin/activate

echo "Mise a jour de pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "Installation des dependances depuis requirements.txt..."
    pip install -r requirements.txt
else
    echo "Erreur : Fichier requirements.txt introuvable !"
    exit 1
fi

echo "Detection du materiel pour PyTorch..."

pip uninstall torch torchvision -y -q 2>/dev/null || true

if command -v nvidia-smi &> /dev/null; then
    echo "GPU NVIDIA detecte."
    echo "Installation de PyTorch (CUDA 12.1)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

elif [ -e /dev/kfd ]; then
    echo "GPU AMD (ROCm) detecte."
    echo "Installation de PyTorch (ROCm 6.2)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

else
    echo "Aucun GPU detecte (ou non supporte)."
    echo "Installation de PyTorch (Version CPU)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "Verification de l'installation..."
python3 -c "import torch; print(f'PyTorch installe : {torch.__version__}'); print(f'Device disponible : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}')"

echo "--- Installation terminee avec succes ---"
echo "Pour activer l'environnement : source .venv/bin/activate"
