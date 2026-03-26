"""
Evaluación paralela de SUPREME sobre múltiples nodos.

Particiona los conjuntos ID y OOD en chunks y asigna cada uno a un nodo.
Cada nodo guarda sus scores en:
    <save_dir>/scores_node_<chunk_id>.pt

El .pt de cada nodo tiene la estructura:
    {
        "id": {
            "scores":    Tensor,  # (N_id_chunk,)
            "id_labels": Tensor,  # (N_id_chunk,)  etiquetas de clase verdaderas
        },
        "<ood_name>": {
            "scores": Tensor,     # (N_ood_chunk,)
        },
        ...
    }

Uso:
    python evaluate_parallel.py \
        --config config/supreme_default.yaml \
        --ckpt   models/supreme/supreme_default/trial0.pth \
        --save_dir results/supreme/supreme_default/node_scores \
        --num_chunks 8 \
        --chunk_id   0
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from supreme.config import Config
from supreme.data import build_support_loader
from supreme.model import SUPREME
from supreme.prototypes import compute_image_prototypes, compute_text_prototypes
from supreme.scores import s_gmp, s_mmp, s_mcm
from utils.datasets import CLIPDataset, load_dataset_from_config
from utils.transforms import default_val_transform


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos del evaluador paralelo.

    Retorna
    -------
    argparse.Namespace : Objeto con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="SUPREME parallel evaluator")
    parser.add_argument("--config",     required=True, help="YAML de evaluación (datasets, score)")
    parser.add_argument("--ckpt",       required=True, help="Ruta al checkpoint .pth")
    parser.add_argument("--save_dir",   required=True, help="Directorio donde guardar scores_node_N.pt")
    parser.add_argument("--num_chunks", type=int, default=1, help="Número total de nodos/chunks")
    parser.add_argument("--chunk_id",   type=int, default=0, help="ID de este nodo (0 … num_chunks-1)")
    parser.add_argument("--batch_size",type=int,default=128)
    return parser.parse_args()


# ── Wrapper para DataParallel ─────────────────────────────────────────────────

class ScoringWrapper(nn.Module):
    """
    Envuelve SUPREME para que DataParallel pueda repartir el batch entre GPUs.
    Devuelve (img_emb, I_prime) por batch.
    """
    def __init__(self, model: SUPREME):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_emb = self.model.encode_image(images)
        I_prime = torch.nn.functional.normalize(self.model.itc.f_img_txt(img_emb), dim=-1)
        return img_emb, I_prime


# ── Chunking ──────────────────────────────────────────────────────────────────

def get_chunk_indices(total: int, num_chunks: int, chunk_id: int) -> list[int]:
    """Reparte los índices por módulo: índice i → nodo i % num_chunks."""
    return [i for i in range(total) if i % num_chunks == chunk_id]


# ── Loader de un chunk ────────────────────────────────────────────────────────

def build_chunk_loader(
    dataset_cfg,
    ckpt_cfg: Config,
    chunk_id: int,
    num_chunks: int,
    batch_size: int,
    transform=None,
) -> DataLoader:
    """
    Construye un DataLoader para el chunk asignado a este nodo.

    Carga el dataset completo según dataset_cfg, selecciona los índices
    correspondientes a chunk_id y retorna un DataLoader sobre ese subconjunto.

    Parámetros
    ----------
    dataset_cfg : Configuración del dataset (source, split, samples_per_class).
    ckpt_cfg    : Configuración del checkpoint (img_size, normalize_*, num_workers).
    chunk_id    : Índice de este nodo (0 … num_chunks-1).
    num_chunks  : Número total de nodos.
    batch_size  : Tamaño del lote.
    transform   : Transformación personalizada; si es None usa default_val_transform.

    Retorna
    -------
    DataLoader : Cargador del chunk para este nodo.
    """
    hf_ds = load_dataset_from_config(
        dataset_cfg.source,
        split=dataset_cfg.split,
        samples_per_class=dataset_cfg.samples_per_class,
    )
    transform = transform or default_val_transform(
        ckpt_cfg.img_size,
        tuple(ckpt_cfg.normalize_mean),
        tuple(ckpt_cfg.normalize_std),
    )
    full_ds = CLIPDataset(hf_ds, transform=transform)
    indices  = get_chunk_indices(len(full_ds), num_chunks, chunk_id)
    subset   = Subset(full_ds, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ckpt_cfg.num_workers,
        pin_memory=True,
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_loader(
    wrapper: nn.Module,
    loader: DataLoader,
    txt_proto: torch.Tensor,
    img_proto: torch.Tensor,
    device: torch.device,
    score_type: str,
    ckpt_cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula las puntuaciones OOD y etiquetas verdaderas para un DataLoader.

    Parámetros
    ----------
    wrapper    : Modelo SUPREME envuelto en ScoringWrapper (opcionalmente DataParallel).
    loader     : DataLoader del conjunto a puntuar.
    txt_proto  : Prototipos de texto normalizados, forma (C, D).
    img_proto  : Prototipos de imagen normalizados, forma (C, D) o (C, K, D).
    device     : Dispositivo de cómputo.
    score_type : Tipo de puntuación: 'gmp', 'mmp', 'mcm_txt' o 'mcm_img'.
    ckpt_cfg   : Configuración del checkpoint (se usa ckpt_cfg.tau).

    Retorna
    -------
    tuple[Tensor, Tensor] : Tensores (scores, labels) en CPU, forma (N,) cada uno.
    """
    wrapper.eval()
    all_scores, all_labels = [], []

    for images, labels in tqdm(loader, desc="Scoring", leave=False):
        images  = images.to(device)
        img_emb, I_prime = wrapper(images)

        if score_type == "gmp":
            sc = s_gmp(img_emb, I_prime, txt_proto, img_proto, ckpt_cfg.tau)
        elif score_type == "mmp":
            sc = s_mmp(img_emb, txt_proto, img_proto, ckpt_cfg.tau)
        elif score_type == "mcm_txt":
            sc = s_mcm(img_emb, txt_proto, ckpt_cfg.tau)
        elif score_type == "mcm_img":
            sc = s_mcm(img_emb, img_proto, ckpt_cfg.tau)
        else:
            raise ValueError(f"Tipo de puntuación desconocido: {score_type}")

        all_scores.append(sc.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores), torch.cat(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Punto de entrada del evaluador paralelo de SUPREME.

    Carga el modelo desde el checkpoint indicado, calcula los prototipos
    (desde el checkpoint si están cacheados, o desde el conjunto de soporte),
    y puntúa el chunk ID y los chunks OOD asignados a este nodo. Los resultados
    se guardan en <save_dir>/scores_node_<chunk_id>.pt. Si el archivo ya existe,
    el nodo termina inmediatamente sin recomputar.
    """
    args = parse_args()

    save_path = os.path.join(args.save_dir, f"scores_node_{args.chunk_id}.pt")
    if os.path.exists(save_path):
        print(f"--- Node {args.chunk_id}: {save_path} ya existe. SKIPPING. ---")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    cfg    = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Node {args.chunk_id}/{args.num_chunks} | device: {device} ---")

    # ── Modelo ────────────────────────────────────────────────────────────────
    ckpt     = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["cfg"]
    class_names: list[str] = ckpt["class_names"]

    model = SUPREME(ckpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    preprocess = model.get_preprocess()
    model.eval()

    num_gpus = torch.cuda.device_count()
    total_batch_size = args.batch_size * max(num_gpus, 1)
    wrapper = ScoringWrapper(model)
    if num_gpus > 1:
        wrapper = nn.DataParallel(wrapper)
        print(f"DataParallel: {num_gpus} GPUs | batch_size {args.batch_size} × {num_gpus} = {total_batch_size}")
    print(f"Modelo cargado: '{args.ckpt}' | {len(class_names)} clases")

    # ── Prototipos ────────────────────────────────────────────────────────────
    if "txt_proto" in ckpt and "img_proto" in ckpt:
        txt_proto = ckpt["txt_proto"].to(device)
        img_proto = ckpt["img_proto"].to(device)
        print("Prototipos cargados desde el checkpoint.")
    else:
        print("Calculando prototipos desde el conjunto de soporte …")
        support_loader, _ = build_support_loader(
            cfg.id_train,
            shots=ckpt_cfg.shots,
            batch_size=ckpt_cfg.batch_size,
            image_size=ckpt_cfg.img_size,
            mean=tuple(ckpt_cfg.normalize_mean),
            std=tuple(ckpt_cfg.normalize_std),
            seed=ckpt_cfg.seed,
            num_workers=ckpt_cfg.num_workers,
        )
        img_proto = compute_image_prototypes(model, support_loader, device, len(class_names))
        txt_proto = compute_text_prototypes(model, class_names, device, ref_loader=support_loader)

    if cfg.mean_img_prototypes and img_proto.dim() == 3:
        img_proto = torch.nn.functional.normalize(img_proto.mean(dim=1), dim=-1)  # (C, D)
        print("img_proto promediado → (C, D)")
    else:
        print(f"img_proto shape: {tuple(img_proto.shape)}")

    results = {}
    # ── ID Val chunk ──────────────────────────────────────────────────────────
    print(f"--- Node {args.chunk_id}: scoring ID Val chunk ---")
    val_loader = build_chunk_loader(cfg.id_val, ckpt_cfg, args.chunk_id, args.num_chunks, total_batch_size, transform=preprocess)
    all_preds, all_val_labels = [], []
    wrapper.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val Acc", leave=False):
            img_emb, _ = wrapper(images.to(device))
            sim = img_emb @ txt_proto.T
            all_preds.append(sim.argmax(dim=1).cpu())
            all_val_labels.append(labels)
    preds = torch.cat(all_preds)
    val_labels = torch.cat(all_val_labels)
    results["id_val"] = {"preds": preds, "labels": val_labels}
    acc = (preds == val_labels).float().mean().item()
    print(f"  ID Val → {len(val_labels)} muestras, acc={acc:.4f}")

    # ── Chunk ID ──────────────────────────────────────────────────────────────
    print(f"--- Node {args.chunk_id}: scoring ID chunk ---")
    id_loader = build_chunk_loader(cfg.id_test, ckpt_cfg, args.chunk_id, args.num_chunks, total_batch_size,transform=preprocess)
    id_scores, id_labels = score_loader(
        wrapper, id_loader, txt_proto, img_proto, device, cfg.score, ckpt_cfg
    )
    results["id"] = {"scores": id_scores, "id_labels": id_labels}
    print(f"  ID → {len(id_scores)} muestras")

    # ── Chunks OOD ────────────────────────────────────────────────────────────
    for ood_cfg in cfg.ood_test:
        name = ood_cfg.source.rstrip("/").split("/")[-1]
        print(f"--- Node {args.chunk_id}: scoring OOD '{name}' chunk ---")
        ood_loader = build_chunk_loader(ood_cfg, ckpt_cfg, args.chunk_id, args.num_chunks, total_batch_size,transform=preprocess)
        ood_scores, _ = score_loader(
            wrapper, ood_loader, txt_proto, img_proto, device, cfg.score, ckpt_cfg
        )
        results[name] = {"scores": ood_scores}
        print(f"  OOD [{name}] → {len(ood_scores)} muestras")

    torch.save(results, save_path)
    print(f"--- Node {args.chunk_id}: guardado en {save_path} ---")


if __name__ == "__main__":
    main()
