"""
Baselines de detección OOD Zero-Shot: MCM y MSP.

Usa el backbone CLIP crudo (sin fine-tuning SUPREME) con prototipos de texto
zero-shot obtenidos con la plantilla "a photo of a {class_name}".

  MCM (Maximum Concept Matching): max(softmax(sim / tau))
  MSP (Maximum Softmax Probability): max(softmax(sim * logit_scale))

Guarda los scores en:
    <save_dir>/scores_node_<chunk_id>.pt

Estructura del .pt guardado:
    {
        "mcm": {
            "id":           {"scores": Tensor, "id_labels": Tensor},
            "<ood_name>":   {"scores": Tensor},
            ...
        },
        "msp": {
            "id":           {"scores": Tensor, "id_labels": Tensor},
            "<ood_name>":   {"scores": Tensor},
            ...
        },
    }

Uso:
    python -m supreme.baselines \
        --config   config/supreme_default.yaml \
        --save_dir results/baselines/supreme_default \
        --num_chunks 8 \
        --chunk_id   0
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from supreme.config import Config
from supreme.scores import s_mcm, s_mmp
from utils.datasets import CLIPDataset, load_dataset_from_config


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos de las baselines.

    Retorna
    -------
    argparse.Namespace : Objeto con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="OOD Baselines: MCM zero-shot & MSP")
    parser.add_argument("--config",     required=True, help="YAML de evaluación (datasets, tau)")
    parser.add_argument("--save_dir",   required=True, help="Directorio donde guardar scores_node_N.pt")
    parser.add_argument("--num_chunks", type=int, default=1, help="Número total de nodos/chunks")
    parser.add_argument("--chunk_id",   type=int, default=0, help="ID de este nodo (0 … num_chunks-1)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tau",        type=float, default=None,
                        help="Temperatura para MCM. Si no se especifica, usa tau del config.")
    parser.add_argument("--ckpt",       nargs="+", default=None,
                        help="Rutas a uno o más checkpoints .pth de SUPREME para cargar img_proto y calcular MMP.")
    return parser.parse_args()


# ── Wrapper para DataParallel ─────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """Envuelve el backbone CLIP para que DataParallel reparta el batch entre GPUs."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.clip.encode_image(images).float(), dim=-1)  # (B, D)


# ── Chunking ──────────────────────────────────────────────────────────────────

def get_chunk_indices(total: int, num_chunks: int, chunk_id: int) -> list[int]:
    """Reparte los índices por módulo: índice i → nodo i % num_chunks."""
    return [i for i in range(total) if i % num_chunks == chunk_id]


def build_chunk_loader(
    dataset_cfg,
    transform,
    num_workers: int,
    chunk_id: int,
    num_chunks: int,
    batch_size: int,
) -> DataLoader:
    """
    Construye un DataLoader para el chunk asignado a este nodo.

    Carga el dataset completo según dataset_cfg, selecciona los índices
    correspondientes a chunk_id y retorna un DataLoader sobre ese subconjunto.

    Parámetros
    ----------
    dataset_cfg : Configuración del dataset (source, split, samples_per_class).
    transform   : Transformación de imagen a aplicar (preprocesado de CLIP).
    num_workers : Número de procesos de carga paralela.
    chunk_id    : Índice de este nodo (0 … num_chunks-1).
    num_chunks  : Número total de nodos.
    batch_size  : Tamaño del lote.

    Retorna
    -------
    DataLoader : Cargador del chunk para este nodo.
    """
    hf_ds = load_dataset_from_config(
        dataset_cfg.source,
        split=dataset_cfg.split,
        samples_per_class=dataset_cfg.samples_per_class,
    )
    full_ds = CLIPDataset(hf_ds, transform=transform)
    indices  = get_chunk_indices(len(full_ds), num_chunks, chunk_id)
    subset   = Subset(full_ds, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ── Prototipos zero-shot ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_zeroshot_text_prototypes(
    clip_model,
    tokenizer,
    class_names: list[str],
    device: torch.device,
    template: str = "{}",
) -> torch.Tensor:
    """Prototipos de texto zero-shot normalizados. Retorna (C, D)."""
    texts  = [template.format(c) for c in class_names]
    tokens = tokenizer(texts).to(device)
    txt_embs = clip_model.encode_text(tokens).float()
    return F.normalize(txt_embs, dim=-1)


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_loader(
    encoder: nn.Module,
    loader: DataLoader,
    txt_proto: torch.Tensor,
    device: torch.device,
    tau: float,
    logit_scale: float,
    mmp_protos: dict[str, torch.Tensor] | None = None,
    # mmp_protos: {"mmp_mean_{k}_shot": Tensor(C,D), "mmp_3d_{k}_shot": Tensor(C,K,D), ...}
) -> dict[str, torch.Tensor]:
    """
    Calcula MCM, MSP y (opcionalmente) MMP sobre un DataLoader.

    Retorna
    -------
    Dict con claves "mcm", "msp", "labels", y una entrada por cada clave en mmp_protos.
    Todos los tensores en CPU, shape (N,).
    """
    encoder.eval()
    accum: dict[str, list] = {"mcm": [], "msp": [], "labels": []}
    for key in (mmp_protos or {}):
        accum[key] = []

    for images, labels in tqdm(loader, desc="Scoring", leave=False):
        images  = images.to(device)
        img_emb = encoder(images)                          # (B, D) normalizado

        sim = img_emb @ txt_proto.T                        # (B, C) similitud coseno

        # MCM: temperatura manual pequeña (alta discriminabilidad)
        accum["mcm"].append(F.softmax(sim / tau,         dim=-1).max(dim=-1).values.cpu())

        # MSP: temperatura nativa de CLIP (logit_scale)
        accum["msp"].append(F.softmax(sim * logit_scale, dim=-1).max(dim=-1).values.cpu())

        # MMP por cada checkpoint cargado
        for key, proto in (mmp_protos or {}).items():
            accum[key].append(s_mmp(img_emb, txt_proto, proto, tau).cpu())

        accum["labels"].append(labels.cpu())

    return {k: torch.cat(v) for k, v in accum.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Punto de entrada de las baselines OOD zero-shot.

    Carga el backbone CLIP crudo (sin fine-tuning SUPREME), calcula prototipos
    de texto zero-shot con la plantilla "a photo of a {class_name}", y puntúa
    el chunk ID Val, ID y los chunks OOD asignados a este nodo con MCM y MSP
    (y opcionalmente MMP si se pasan checkpoints). Los resultados se guardan en
    <save_dir>/scores_node_<chunk_id>.pt. Si el archivo ya existe, el nodo
    termina inmediatamente sin recomputar.
    """
    args = parse_args()

    save_path = os.path.join(args.save_dir, f"scores_node_{args.chunk_id}.pt")
    if os.path.exists(save_path):
        print(f"--- Node {args.chunk_id}: {save_path} ya existe. SKIPPING. ---")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    cfg    = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau    = args.tau if args.tau is not None else cfg.tau
    print(f"--- Node {args.chunk_id}/{args.num_chunks} | device: {device} | tau={tau} ---")

    # ── Modelo CLIP crudo ──────────────────────────────────────────────────────
    clip_model, _, preprocess = open_clip.create_model_and_transforms(cfg.clip_model,pretrained=cfg.clip_pretrained)
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer(cfg.clip_model)

    logit_scale = clip_model.logit_scale.exp().item()
    print(f"Modelo: '{cfg.clip_model}' | logit_scale={logit_scale:.4f} | tau={tau}")

    # ── Nombres de clase desde el ID dataset ──────────────────────────────────
    hf_id       = load_dataset_from_config(cfg.id_test.source, split=cfg.id_test.split)
    class_names = CLIPDataset(hf_id).class_names
    print(f"Clases: {len(class_names)}")

    # ── Prototipos de texto zero-shot ─────────────────────────────────────────
    txt_proto = compute_zeroshot_text_prototypes(clip_model, tokenizer, class_names, device)
    print(f"Prototipos de texto: {txt_proto.shape}")

    # ── Prototipos de imagen desde checkpoints (opcional) ─────────────────────
    mmp_protos: dict[str, torch.Tensor] = {}
    if args.ckpt is not None:
        for ckpt_path in args.ckpt:
            ckpt         = torch.load(ckpt_path, map_location=device, weights_only=False)
            img_proto_3d = ckpt["img_proto"].to(device)                          # (C, K, D)
            k            = img_proto_3d.shape[1]
            mmp_protos[f"mmp_mean_{k}_shot"] = F.normalize(img_proto_3d.mean(dim=1), dim=-1)  # (C, D)
            mmp_protos[f"mmp_3d_{k}_shot"]   = img_proto_3d                                   # (C, K, D)
            print(f"img_proto cargado desde '{ckpt_path}' | K={k} | shape: {tuple(img_proto_3d.shape)}")

    # ── Encoder (con DataParallel si hay múltiples GPUs) ──────────────────────
    num_gpus         = torch.cuda.device_count()
    total_batch_size = args.batch_size * max(num_gpus, 1)
    encoder          = ImageEncoder(clip_model)
    if num_gpus > 1:
        encoder = nn.DataParallel(encoder)
        print(f"DataParallel: {num_gpus} GPUs | batch_size {args.batch_size} × {num_gpus} = {total_batch_size}")

    score_keys = ["mcm", "msp", *mmp_protos.keys()]
    results = {k: {} for k in score_keys}

    # ── ID Val chunk ───────────────────────────────────────────────────────
    
    print(f"--- Node {args.chunk_id}: scoring ID Val chunk ---")
    val_loader = build_chunk_loader(
        cfg.id_val, preprocess, cfg.num_workers,
        args.chunk_id, args.num_chunks, total_batch_size,
    )
    all_preds, all_labels = [], []
    encoder.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val Acc", leave=False):
            sim = encoder(images.to(device)) @ txt_proto.T
            all_preds.append(sim.argmax(dim=1).cpu())
            all_labels.append(labels)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    results["id_val"] = {"preds": preds, "labels": labels}
    acc = (preds == labels).float().mean().item()
    print(f"  ID Val → {len(labels)} muestras, acc={acc:.4f}")
    # ── ID chunk ───────────────────────────────────────────────────────────────
    print(f"--- Node {args.chunk_id}: scoring ID chunk ---")
    id_loader = build_chunk_loader(
        cfg.id_test, preprocess, cfg.num_workers,
        args.chunk_id, args.num_chunks, total_batch_size,
    )
    id_out = score_loader(encoder, id_loader, txt_proto, device, tau, logit_scale, mmp_protos or None)
    id_labels = id_out["labels"]
    for key in score_keys:
        results[key]["id"] = {"scores": id_out[key], "id_labels": id_labels}
    print(f"  ID → {len(id_labels)} muestras")

    # ── OOD chunks ────────────────────────────────────────────────────────────
    for ood_cfg in cfg.ood_test:
        name = ood_cfg.source.rstrip("/").split("/")[-1]
        print(f"--- Node {args.chunk_id}: scoring OOD '{name}' chunk ---")
        ood_loader = build_chunk_loader(
            ood_cfg, preprocess, cfg.num_workers,
            args.chunk_id, args.num_chunks, total_batch_size,
        )
        ood_out = score_loader(encoder, ood_loader, txt_proto, device, tau, logit_scale, mmp_protos or None)
        for key in score_keys:
            results[key][name] = {"scores": ood_out[key]}
        print(f"  OOD [{name}] → {len(ood_out['labels'])} muestras")

    torch.save(results, save_path)
    print(f"--- Node {args.chunk_id}: guardado en {save_path} ---")


if __name__ == "__main__":
    main()
