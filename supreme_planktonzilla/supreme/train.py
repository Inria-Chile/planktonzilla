"""
Entrenamiento del método SUPREME.

Los modelos se guardan en:
    models/supreme/<nombre_config>/trial<N>.pth

Los logs de entrenamiento se guardan en:
    models/supreme/<nombre_config>/logs/trial<N>.log

Se utiliza como módulo desde main.py mediante la función run(), o bien
directamente como script:
    python -m supreme.train --config config/supreme_default.yaml
"""

import os
import random

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from .config import Config
from .data import build_support_loader
from .losses import total_loss
from .model import SUPREME
from .prototypes import compute_image_prototypes, compute_text_prototypes
from utils.io import config_stem
from utils.logger import ExperimentLogger


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Fija todas las semillas aleatorias para reproducibilidad del experimento.

    Args:
        seed (int): Valor entero de la semilla.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_trainable_params(model: SUPREME) -> list:
    """
    Retorna únicamente los parámetros del modelo que requieren gradiente.

    Args:
        model (SUPREME): Modelo SUPREME con backbone congelado.

    Returns:
        list: Lista de parámetros entrenables (BPG e ITC).
    """
    return [p for p in model.parameters() if p.requires_grad]


# ── Bucle de entrenamiento ────────────────────────────────────────────────────

def train_one_epoch(
    model: SUPREME,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    class_names: list[str],
    device: torch.device,
    cfg: Config,
    epoch: int,
) -> dict:
    """
    Ejecuta una época completa de entrenamiento con precisión mixta (AMP).

    Args:
        model       : Modelo SUPREME en modo entrenamiento.
        loader      : DataLoader del conjunto de soporte.
        optimizer   : Optimizador SGD configurado sobre los parámetros entrenables.
        scaler      : GradScaler para el escalado de gradientes con AMP.
        class_names : Lista de nombres de clase del conjunto ID.
        device      : Dispositivo de cómputo.
        cfg         : Configuración del experimento.
        epoch       : Índice de la época actual (para mostrar progreso).

    Returns:
        dict: Promedio de cada componente de la pérdida sobre todos los lotes.
    """
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            fwd = model(images, class_names, labels)
            loss, components = total_loss(fwd, cfg, model.bpg.mu)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ── Trial ─────────────────────────────────────────────────────────────────────

def run_trial(
    cfg: Config,
    trial: int,
    out_dir: str,
    log_dir: str,
) -> str:
    """
    Ejecuta un trial completo: carga de datos, entrenamiento, cómputo de
    prototipos y guardado del checkpoint.

    Crea un logger propio para el trial con su archivo .log correspondiente.

    Args:
        cfg     : Configuración del experimento.
        trial   : Índice del trial (0-based).
        out_dir : Directorio donde se guardará el checkpoint.
        log_dir : Directorio donde se guardará el archivo .log del trial.

    Returns:
        str: Ruta al checkpoint guardado.
    """
    seed = cfg.seed + trial
    set_seed(seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Logger propio del trial con archivo de salida
    log_path = os.path.join(log_dir, f"trial{trial}.log")
    logger = ExperimentLogger(name=f"supreme.train.trial{trial}")
    logger.add_file_handler(log_path)

    logger.info(f"{'='*60}")
    logger.info(f"Trial {trial + 1}/{cfg.trials}  |  seed={seed}  |  device={device}")
    logger.info(f"{'='*60}")
        # ── Modelo ────────────────────────────────────────────────────────────────
    model = SUPREME(cfg).to(device)
    trainable = get_trainable_params(model)
    n_params = sum(p.numel() for p in trainable)
    preprocess = model.get_preprocess()
    logger.info(f"Parámetros entrenables: {n_params:,}  (BPG + ITC)")

    # ── Datos ─────────────────────────────────────────────────────────────────
    logger.start_timer("data_loading")
    support_loader, class_names = build_support_loader(
        cfg.id_train,
        shots=cfg.shots,
        batch_size=cfg.batch_size,
        transform=preprocess,
        seed=seed,
        num_workers=cfg.num_workers,
        logger=logger,
    )
    logger.end_timer("data_loading")
    logger.info(
        f"Clases ({len(class_names)}): "
        f"{class_names[:5]}{'...' if len(class_names) > 5 else ''}"
    )

    img_proto_loader, _ = build_support_loader(
        cfg.id_train,
        shots=None,
        batch_size=cfg.batch_size,
        transform=preprocess,
        seed=seed,
        num_workers=cfg.num_workers,
        logger=logger,
    )

    # ── Optimizador y GradScaler (AMP) ────────────────────────────────────────
    # Grupo separado para mu y sigma del sesgo gaussiano (lr reducido para evitar explosión)
    bias_params = list(model.bpg.mu.unsqueeze(0)) + list(model.bpg.sigma.unsqueeze(0))
    bias_param_ids = {id(model.bpg.mu), id(model.bpg.sigma)}
    other_params = [p for p in trainable if id(p) not in bias_param_ids]
    param_groups = [
        {"params": other_params,                          "lr": cfg.lr},
        {"params": [model.bpg.mu, model.bpg.sigma],      "lr": cfg.lr_bias},
    ]
    optimizer = torch.optim.SGD(param_groups, momentum=cfg.momentum)
    scaler = amp.GradScaler()
    logger.info(
        f"Optimizador: SGD  |  lr={cfg.lr}  |  lr_bias={cfg.lr_bias}  |  momentum={cfg.momentum}"
    )
    logger.info(f"Épocas: {cfg.epochs}  |  batch_size: {cfg.batch_size}  |  AMP: activado")

    # ── Bucle de entrenamiento ────────────────────────────────────────────────
    logger.start_timer("training")
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        metrics = train_one_epoch(
            model, support_loader, optimizer, scaler, class_names, device, cfg, epoch
        )
        history.append(metrics)
        if epoch % 10 == 0 or epoch == 1:
            parts = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"[Epoch {epoch:3d}/{cfg.epochs}]  {parts}")
    logger.end_timer("training")

    # ── Cómputo de prototipos ─────────────────────────────────────────────────
    model.eval()
    logger.start_timer("prototype_computation")
    logger.info("Calculando prototipos de imagen y texto …")
    img_proto = compute_image_prototypes(
        model, support_loader, device, num_classes=len(class_names)
    )
    txt_proto = compute_text_prototypes(
        model, class_names, device, ref_loader=support_loader
    )
    logger.end_timer("prototype_computation")
    logger.info(
        f"Prototipos calculados  |  img: {tuple(img_proto.shape)}  "
        f"|  txt: {tuple(txt_proto.shape)}"
    )

    # ── Guardado del checkpoint ───────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, f"trial{trial}.pth")
    torch.save(
        {
            "model_state": model.state_dict(),
            "img_proto": img_proto.cpu(),
            "txt_proto": txt_proto.cpu(),
            "class_names": class_names,
            "cfg": cfg,
            "trial": trial,
            "seed": seed,
        },
        ckpt_path,
    )

    # ── Guardado de pérdidas ──────────────────────────────────────────────────
    losses_path = os.path.join(out_dir, f"trial{trial}_losses.pt")
    loss_keys = history[0].keys()
    torch.save(
        {k: torch.tensor([e[k] for e in history]) for k in loss_keys},
        losses_path,
    )

    logger.info(f"Checkpoint guardado → {ckpt_path}")
    logger.info(f"Pérdidas guardadas  → {losses_path}")
    logger.info(f"Log guardado        → {log_path}")
    return ckpt_path


# ── Punto de entrada principal ────────────────────────────────────────────────

def run(config_path: str) -> str:
    """
    Punto de entrada para el entrenamiento de SUPREME.

    Carga la configuración desde el YAML indicado, ejecuta todos los trials
    y guarda los checkpoints bajo models/supreme/<nombre_config>/ y los logs
    bajo models/supreme/<nombre_config>/logs/.

    Args:
        config_path (str): Ruta al archivo YAML de configuración.

    Returns:
        str: Directorio donde se guardaron los modelos del experimento.
    """
    cfg = Config.from_yaml(config_path)
    name = config_stem(config_path)

    out_dir = os.path.join("models", "supreme", name)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Logger de nivel superior (solo consola) para el resumen global
    top_logger = ExperimentLogger(name=f"supreme.train.{name}")
    top_logger.info(f"{'='*60}")
    top_logger.info(f"SUPREME Training  |  config: {name}  |  trials: {cfg.trials}")
    top_logger.info(f"Checkpoints → {out_dir}")
    top_logger.info(f"Logs        → {log_dir}")
    top_logger.info(f"{'='*60}")

    top_logger.start_timer("total_training")
    ckpt_paths = []
    for trial in range(cfg.trials):
        path = run_trial(cfg, trial, out_dir, log_dir)
        ckpt_paths.append(path)
    top_logger.end_timer("total_training")

    top_logger.info("=== Entrenamiento completado ===")
    for p in ckpt_paths:
        top_logger.info(f"  {p}")

    return out_dir


# ── Ejecución directa como script ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar SUPREME directamente")
    parser.add_argument("--config", required=True,
                        help="Ruta al archivo YAML de configuración")
    args = parser.parse_args()
    run(args.config)
