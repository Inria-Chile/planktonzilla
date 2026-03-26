"""
Selección inteligente de ejemplos Few-Shot mediante K-Means Intra-Clase.
Extrae embeddings usando BioCLIP y selecciona las K imágenes más cercanas
a los centroides de cada clase para maximizar la diversidad morfológica.
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from supreme.config import Config
from supreme.model import SUPREME
from utils.datasets import CLIPDataset, load_dataset_from_config
from utils.transforms import default_val_transform


# ── Wrapper para DataParallel ─────────────────────────────────────────────────
class EmbeddingWrapper(nn.Module):
    """
    Envuelve el método encode_image para que DataParallel 
    pueda repartir el batch entre múltiples GPUs a través de forward().
    """
    def __init__(self, model: SUPREME):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)


def parse_args():
    parser = argparse.ArgumentParser(description="K-Means Few-Shot Selection")
    parser.add_argument("--config", required=True, help="Ruta al YAML (ej. supreme_vitb16_bioclip3.yaml)")
    parser.add_argument("--shots", type=int, default=16, help="Número de ejemplos por clase (K)")
    parser.add_argument("--out", type=str, default="/home/svasquez/clip_prompt_learning_planktonzilla/data/planktonzilla_16shot_indices.json", help="Archivo de salida")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando K-Means Selection ({args.shots}-shot) ---")

    # 1. Cargar el modelo SUPREME (Solo necesitamos el backbone congelado)
    model = SUPREME(cfg).to(device)
    preprocess = model.get_preprocess()
    model.eval()

    # 1.5 Configurar DataParallel
    num_gpus = torch.cuda.device_count()
    wrapper = EmbeddingWrapper(model)
    
    if num_gpus > 1:
        print(f"Activando DataParallel con {num_gpus} GPUs.")
        wrapper = nn.DataParallel(wrapper)
        # Escalar el batch size según las GPUs disponibles
        total_batch_size = args.batch_size * num_gpus
    else:
        print(f"Usando un solo dispositivo: {device}")
        total_batch_size = args.batch_size

    # 2. Cargar el dataset de entrenamiento COMPLETO (sin recorte de shots aún)
    hf_ds = load_dataset_from_config(cfg.id_train.source, split=cfg.id_train.split)
    
    # Usamos la transformación de validación para extraer características puras (sin data augmentation)
    transform = preprocess or default_val_transform(
        cfg.img_size, tuple(cfg.normalize_mean), tuple(cfg.normalize_std)
    )
    dataset = CLIPDataset(hf_ds, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=total_batch_size, # Usar el batch size escalado
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 3. Extraer todos los embeddings
    print(f"Extrayendo embeddings (Batch Size efectivo: {total_batch_size})...")
    all_embs = []
    all_labels = []
    
    # Asegurarse de que el wrapper esté en eval()
    wrapper.eval()
    
    for images, labels in tqdm(loader, desc="Embeddings"):
        images = images.to(device)
        # Usamos el wrapper en lugar de model.encode_image directamente
        emb = wrapper(images)
        all_embs.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    num_classes = len(np.unique(all_labels))
    print(f"Total imágenes: {len(all_embs)} | Total clases: {num_classes}")

    # 4. Aplicar K-Means por clase
    selected_indices = []
    print(f"Ejecutando K-Means (K={args.shots}) intra-clase...")
    
    for c in tqdm(range(num_classes), desc="Clustering"):
        # Obtener índices globales de las imágenes que pertenecen a esta clase
        class_idx = np.where(all_labels == c)[0]
        
        # Si una clase tiene menos imágenes que los shots pedidos, tomamos todas
        if len(class_idx) <= args.shots:
            selected_indices.extend(class_idx.tolist())
            continue
            
        class_embs = all_embs[class_idx]
        
        # Ejecutar K-Means
        kmeans = KMeans(n_clusters=args.shots, random_state=cfg.seed, n_init=10)
        kmeans.fit(class_embs)
        
        # Calcular distancias de todos los puntos de la clase a los centroides
        distances = pairwise_distances(class_embs, kmeans.cluster_centers_)
        
        # Encontrar el índice (local a la clase) más cercano a cada centroide
        closest_local_idx = np.argmin(distances, axis=0)
        
        # Mapear de vuelta a los índices globales del dataset original
        closest_global_idx = class_idx[closest_local_idx]
        selected_indices.extend(closest_global_idx.tolist())

    # 5. Guardar resultados
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"indices": selected_indices}, f)
        
    print(f"\n¡Éxito! {len(selected_indices)} índices guardados en {args.out}")


if __name__ == "__main__":
    main()