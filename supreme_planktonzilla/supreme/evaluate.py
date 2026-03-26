"""
Evaluación del método SUPREME.

Calcula FPR95, AUROC y exactitud Top-1 sobre el conjunto de validación ID
y uno o varios conjuntos OOD.

Los resultados y el log de evaluación se guardan en:
    results/supreme/<nombre_carpeta_modelos>/scores.pt
    results/supreme/<nombre_carpeta_modelos>/evaluation.log

scores.pt contiene un dict anidado con la estructura:
    {
        "<trial_name>": {
            "id": {
                "scores":    Tensor,   # (N_id,)
                "id_labels": Tensor,   # (N_id,)  etiquetas de clase verdaderas
            },
            "<ood_name>": {
                "scores": Tensor,      # (N_ood,)
            },
            ...
        },
        ...
    }

Se utiliza como módulo desde main.py mediante la función run(), o bien
directamente como script:
    python -m supreme.evaluate --model_dir models/supreme/supreme_default
                               --config config/supreme_default.yaml
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .data import build_eval_loader, build_support_loader
from .model import SUPREME
from .prototypes import compute_image_prototypes, compute_text_prototypes
from .scores import s_gmp, s_mmp, s_mcm
from utils.metricas import fpr_at_tpr, auroc
from utils.logger import ExperimentLogger


# ── Extracción de puntuaciones ────────────────────────────────────────────────

@torch.no_grad()
def extract_scores(
    model: SUPREME,
    loader,
    txt_proto: torch.Tensor,
    img_proto: torch.Tensor,
    device: torch.device,
    score_type: str,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae puntuaciones OOD y etiquetas verdaderas para un conjunto de datos.

    Args:
        model      : Modelo SUPREME en modo evaluación.
        loader     : DataLoader del conjunto a puntuar.
        txt_proto  : Prototipos de texto normalizados, forma (C, D).
        img_proto  : Prototipos de imagen normalizados, forma (C, D).
        device     : Dispositivo de cómputo.
        score_type : Tipo de puntuación: 'gmp', 'mmp', 'mcm_txt' o 'mcm_img'.
        cfg        : Configuración del experimento (se usa cfg.tau).

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays (scores, labels).
            scores: Puntuaciones de confianza (mayor = más in-distribution).
            labels: Etiquetas de clase verdaderas.
    """
    model.eval()
    all_scores, all_labels = [], []

    for images, labels in tqdm(loader, desc="Scoring", leave=False):
        images = images.to(device)
        # evaluate.py - extract_scores()
        img_emb = model.encode_image(images)                    # (B, D) normalizado
        I_prime = torch.nn.functional.normalize(model.itc.f_img_txt(img_emb), dim=-1) # (B, D)

        if score_type == "gmp":
            sc = s_gmp(img_emb, I_prime, txt_proto, img_proto, cfg.tau)
        elif score_type == "mmp":
            sc = s_mmp(img_emb, txt_proto, img_proto, cfg.tau)
        elif score_type == "mcm_txt":
            sc = s_mcm(img_emb, txt_proto, cfg.tau)
        elif score_type == "mcm_img":
            sc = s_mcm(img_emb, img_proto, cfg.tau)
        else:
            raise ValueError(f"Tipo de puntuación desconocido: {score_type}")

        all_scores.append(sc.cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_scores).numpy(),
        torch.cat(all_labels).numpy(),
    )



# ── Evaluación de un checkpoint ───────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: str,
    cfg: Config,
    device: torch.device,
    logger: ExperimentLogger,
) -> dict:
    """
    Evalúa un checkpoint individual sobre el conjunto ID y todos los OOD.

    Args:
        ckpt_path : Ruta al archivo .pth del checkpoint.
        cfg       : Configuración con rutas de datos y función de puntuación.
        device    : Dispositivo de cómputo.
        logger    : Logger para registrar métricas y tiempos.

    Returns:
        dict: clave ``"id"`` con scores e id_labels del conjunto ID, y una
              clave por cada dataset OOD con sus scores.
    """
    ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)
    ckpt_cfg: Config = ckpt["cfg"]
    class_names: list[str] = ckpt["class_names"]
    C = len(class_names)

    model = SUPREME(ckpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Modelo cargado desde '{ckpt_path}'  |  {C} clases")

    # ── Prototipos ────────────────────────────────────────────────────────────
    if "txt_proto" in ckpt and "img_proto" in ckpt:
        logger.info("Cargando prototipos cacheados del checkpoint …")
        txt_proto = ckpt["txt_proto"].to(device)
        img_proto = ckpt["img_proto"].to(device)
    else:
        logger.info("Calculando prototipos desde el conjunto de soporte …")
        if not cfg.id_train.source:
            raise ValueError(
                "id_train.source es necesario en el YAML cuando el checkpoint "
                "no tiene prototipos cacheados."
            )
        logger.start_timer("prototype_computation")
        support_loader, _ = build_support_loader(
            cfg.id_train,
            shots=ckpt_cfg.shots,
            batch_size=ckpt_cfg.batch_size,
            image_size=ckpt_cfg.img_size,
            mean=tuple(ckpt_cfg.normalize_mean),
            std=tuple(ckpt_cfg.normalize_std),
            num_workers=ckpt_cfg.num_workers,
            seed=ckpt_cfg.seed,
            logger=logger,
        )
        img_proto = compute_image_prototypes(model, support_loader, device, C)
        txt_proto = compute_text_prototypes(
            model, class_names, device, ref_loader=support_loader
        )
        logger.end_timer("prototype_computation")

    txt_proto = txt_proto.to(device)
    img_proto = img_proto.to(device)

    # ── Conjunto de validación ID ─────────────────────────────────────────────
    logger.start_timer("id_evaluation")
    logger.info(f"Evaluando conjunto ID: '{cfg.id_test.source}' (split='{cfg.id_test.split}')")
    id_loader, _ = build_eval_loader(
        cfg.id_test,
        batch_size=ckpt_cfg.batch_size,
        image_size=ckpt_cfg.img_size,
        mean=tuple(ckpt_cfg.normalize_mean),
        std=tuple(ckpt_cfg.normalize_std),
        num_workers=ckpt_cfg.num_workers,
        logger=logger,
    )
    id_scores, id_labels = extract_scores(
        model, id_loader, txt_proto, img_proto, device, cfg.score, ckpt_cfg
    )
    logger.end_timer("id_evaluation")
    logger.info(f"ID Val  →  {len(id_scores)} muestras")

    # ── Conjuntos OOD ─────────────────────────────────────────────────────────
    ood_results = {}
    for ood_cfg in cfg.ood_test:
        name = ood_cfg.source.rstrip("/").split("/")[-1]
        logger.start_timer(f"ood_{name}")
        logger.info(f"Evaluando OOD: '{ood_cfg.source}' (split='{ood_cfg.split}')")
        ood_loader, _ = build_eval_loader(
            ood_cfg,
            batch_size=ckpt_cfg.batch_size,
            image_size=ckpt_cfg.img_size,
            mean=tuple(ckpt_cfg.normalize_mean),
            std=tuple(ckpt_cfg.normalize_std),
            num_workers=ckpt_cfg.num_workers,
            logger=logger,
        )
        ood_scores, _ = extract_scores(
            model, ood_loader, txt_proto, img_proto, device, cfg.score, ckpt_cfg
        )
        fpr = fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95)
        auc = auroc(id_scores, ood_scores)
        logger.end_timer(f"ood_{name}")
        logger.info(
            f"OOD [{name}]  →  FPR95: {fpr * 100:.2f}%  |  AUROC: {auc * 100:.2f}%"
        )
        ood_results[name] = {"scores": torch.tensor(ood_scores)}

    return {
        "id": {
            "scores":    torch.tensor(id_scores),
            "id_labels": torch.tensor(id_labels),
        },
        **ood_results,
    }


# ── Punto de entrada principal ────────────────────────────────────────────────

def run(model_dir: str, config_path: str) -> None:
    """
    Punto de entrada para la evaluación de SUPREME.

    Evalúa todos los checkpoints encontrados en model_dir usando las rutas
    de datos del config YAML. Los resultados y el log se guardan en:
        results/supreme/<nombre_de_model_dir>/

    Args:
        model_dir   (str): Carpeta donde se guardaron los modelos durante el entrenamiento.
        config_path (str): Ruta al archivo YAML de configuración (rutas de datos y score).
    """
    cfg = Config.from_yaml(config_path)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_name = os.path.basename(model_dir.rstrip("/"))
    results_dir = os.path.join("results", "supreme", run_name)
    os.makedirs(results_dir, exist_ok=True)

    # Logger con archivo de salida para toda la evaluación
    log_path = os.path.join(results_dir, "evaluation.log")
    logger = ExperimentLogger(name=f"supreme.eval.{run_name}")
    logger.add_file_handler(log_path)

    # Buscar todos los checkpoints en el directorio del modelo
    ckpt_files = sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".pth")
    ])
    if not ckpt_files:
        logger.error(f"No se encontraron checkpoints (.pth) en: {model_dir}")
        raise FileNotFoundError(f"No se encontraron checkpoints (.pth) en: {model_dir}")

    logger.info(f"{'='*60}")
    logger.info(f"SUPREME Evaluation  |  modelos: {model_dir}")
    logger.info(f"Score: {cfg.score}  |  device: {device}")
    logger.info(f"Checkpoints encontrados: {len(ckpt_files)}")
    logger.info(f"Resultados → {results_dir}")
    logger.info(f"{'='*60}")

    logger.start_timer("total_evaluation")
    all_results = {}
    for ckpt_path in ckpt_files:
        trial_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        logger.info(f"\n--- Evaluando {trial_name} ---")
        trial_result = evaluate_checkpoint(ckpt_path, cfg, device, logger)
        all_results[trial_name] = trial_result

    logger.end_timer("total_evaluation")

    scores_path = os.path.join(results_dir, "scores.pt")
    torch.save(all_results, scores_path)
    logger.info(f"\nPuntuaciones guardadas → {scores_path}")
    logger.info(f"Log guardado           → {log_path}")


# ── Ejecución directa como script ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluar SUPREME directamente")
    parser.add_argument("--model_dir", required=True,
                        help="Carpeta donde se guardaron los modelos entrenados")
    parser.add_argument("--config", required=True,
                        help="Ruta al archivo YAML de configuración")
    args = parser.parse_args()
    run(args.model_dir, args.config)
