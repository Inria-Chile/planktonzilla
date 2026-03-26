"""
Cómputo de prototipos para el método SUPREME.

Prototipos de imagen : media de clase de los embeddings de imagen normalizados
                       obtenidos del conjunto de soporte.
Prototipos de texto  : salida de f_text(IDBP_c) usando el BPG entrenado en
                       modo de inferencia (b = μ, sin muestreo de ruido).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import SUPREME


@torch.no_grad()
def compute_image_prototypes(
    model: "SUPREME",
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    shots: int = 16, # Añadimos el número de shots (K)
) -> torch.Tensor:
    """
    Calcula los prototipos de imagen preservando cada ejemplo (Multi-Prototipo).
    En lugar de promediar, retorna un tensor 3D con los K ejemplos por clase
    para habilitar la búsqueda del vecino más cercano (k-NN).

    Parámetros
    ----------
    model       : Modelo SUPREME con el backbone congelado.
    loader      : DataLoader del conjunto de soporte (ej. 16-shot K-Means).
    device      : Dispositivo de cómputo (CPU o CUDA).
    num_classes : Número total de clases C.
    shots       : Número de ejemplos por clase (K).

    Retorna
    -------
    Tensor de forma (C, K, D) con los prototipos de imagen normalizados (L2).
    """
    D = model.cfg.embed_dim
    
    # Inicializamos el tensor 3D: (Clases, Shots, Dimensiones)
    protos = torch.zeros(num_classes, shots, D, device=device)
    
    # Contador para saber en qué 'slot' (0 a 15) guardar la siguiente imagen de cada clase
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    for images, labels in tqdm(loader, desc="Image prototypes (3D)", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            feats = model.encode_image(images)   # (B, D) normalizado
            
        # Guardamos cada característica en su respectiva clase y posición
        for i, c in enumerate(labels):
            c_idx = c.item()
            idx = counts[c_idx].item()
            
            if idx < shots:
                protos[c_idx, idx] = feats[i]
                counts[c_idx] += 1

    # Failsafe: Si una clase tiene menos de 'shots' imágenes, clonamos 
    # la última imagen válida para rellenar el tensor.
    for c in range(num_classes):
        valid_count = counts[c].item()
        if 0 < valid_count < shots:
            for missing_idx in range(valid_count, shots):
                protos[c, missing_idx] = protos[c, valid_count - 1]
        elif valid_count == 0:
            print(f"[Warning] La clase {c} no tiene imágenes en el soporte.")

    # Aseguramos que los vectores mantengan magnitud 1
    return F.normalize(protos, dim=-1)       # (C, K, D)


@torch.no_grad()
def compute_text_prototypes(
    model: "SUPREME",
    class_names: list[str],
    device: torch.device,
    ref_loader: DataLoader | None = None,
) -> torch.Tensor:
    """
    Calcula los prototipos de texto usando el BPG entrenado en modo de
    inferencia (b = μ, sin ruido gaussiano).

    Si se proporciona ref_loader, los prototipos se promedian sobre las
    imágenes de referencia para mayor estabilidad. En caso contrario, se usa
    un embedding de imagen cero (solo μ contribuye como sesgo).

    Parámetros
    ----------
    model       : Modelo SUPREME en modo eval con el BPG entrenado.
    class_names : Lista de nombres de clase (C elementos).
    device      : Dispositivo de cómputo.
    ref_loader  : DataLoader de referencia opcional para estabilizar prototipos.

    Retorna
    -------
    Tensor de forma (C, D) con los prototipos de texto normalizados (L2).
    """
    model.eval()
    D = model.cfg.embed_dim
    C = len(class_names)

    if ref_loader is not None:
        # Acumular prototipos de texto sobre un lote de referencia
        all_protos = []
        for images, _ in tqdm(ref_loader, desc="Text prototypes", leave=False):
            images = images.to(device)
            img_emb = model.encode_image(images)              # (B, D)
            proto, _ = model.encode_text_with_bpg(img_emb, class_names)
            all_protos.append(proto.mean(0))                  # (C, D)
        txt_proto = torch.stack(all_protos).mean(0)           # (C, D)
    else:
        # Usar embedding de imagen cero: solo μ contribuye como sesgo
        dummy = torch.zeros(1, D, device=device)
        txt_proto, _ = model.encode_text_with_bpg(dummy, class_names)
        txt_proto = txt_proto.squeeze(0)                      # (C, D)

    return F.normalize(txt_proto, dim=-1)
