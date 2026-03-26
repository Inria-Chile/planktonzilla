"""
Script de optimización de hiperparámetros con Optuna para SUPREME.
Estrategia: Entrena 50 épocas completas, calcula prototipos una vez al final,
y evalúa sobre un "Mini-Val" estratificado para MAXIMIZAR el Validation Accuracy.
"""

import argparse
import random
import logging
from collections import defaultdict

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import torch
import torch.cuda.amp as amp
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from supreme.config import Config
from supreme.data import build_support_loader
from supreme.model import SUPREME
from supreme.train import train_one_epoch, set_seed, get_trainable_params
from supreme.prototypes import compute_text_prototypes
from utils.datasets import load_dataset_from_config, CLIPDataset
from utils.transforms import default_val_transform

# Desactivar logs extensos de Optuna para mantener limpia la consola
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial: optuna.Trial, config_path: str) -> float:
    """
    Función objetivo de Optuna para un trial individual.

    Carga la configuración base desde config_path, sugiere valores para los
    hiperparámetros en el espacio de búsqueda definido, entrena SUPREME durante
    50 épocas completas sobre el conjunto de soporte few-shot, y evalúa la
    exactitud Top-1 sobre un Mini-Val estratificado de 4 muestras por clase.

    Parámetros
    ----------
    trial       : Objeto Trial de Optuna que gestiona el espacio de búsqueda.
    config_path : Ruta al archivo YAML de configuración base del experimento.

    Retorna
    -------
    float : Exactitud de validación final (valor a maximizar).
    """
    # 1. Cargar la configuración base
    cfg = Config.from_yaml(config_path)

    # 2. Espacio de búsqueda protegido (Evitamos memorización extrema)
    cfg.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg.lr_bias = trial.suggest_float("lr_bias", 1e-5, 1e-3, log=True)
    cfg.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) 
    cfg.context_length = trial.suggest_categorical("context_length", [4, 8])
    cfg.tau = trial.suggest_float("tau", 0.015, 0.05) 
    cfg.momentum = trial.suggest_float("momentum", 0.85, 0.99)
    cfg.alpha = trial.suggest_float("alpha", 0.001, 0.005) 

    # Parámetros fijos
    cfg.epochs = 50
    cfg.beta = 0.05
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Logger silenciado para los trials
    class DummyLogger:
        def info(self, msg, *args, **kwargs): pass
        def start_timer(self, name): pass
        def end_timer(self, name): pass
        def add_file_handler(self, path): pass

    dummy_logger = DummyLogger()
    
    # --- Dataloader de Entrenamiento ---
    loader, class_names = build_support_loader(
        cfg.id_train,
        shots=cfg.shots,
        batch_size=cfg.batch_size,
        image_size=cfg.img_size,
        mean=tuple(cfg.normalize_mean),
        std=tuple(cfg.normalize_std),
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        logger=dummy_logger,
    )

    # --- Mini-Val Proxy Estratificado ---
    # --- Mini-Val Proxy Estratificado (4 shots por clase) ---
    hf_val_ds = load_dataset_from_config(cfg.id_val.source, split=cfg.id_val.split)
    
    # Extraer etiquetas según el formato de HuggingFace
    if "label" in hf_val_ds.column_names:
        val_labels = hf_val_ds["label"]
    else:
        val_labels = [item["label"] for item in hf_val_ds]
        
    # Agrupar índices por clase
    class_indices = defaultdict(list)
    for idx, label in enumerate(val_labels):
        class_indices[label].append(idx)
        
    # FIJAMOS EXACTAMENTE 4 MUESTRAS POR CLASE
    samples_per_class = 4
    
    val_indices = []
    rng = random.Random(cfg.seed)
    
    for label, indices in class_indices.items():
        # k será 4, a menos que la clase tenga menos de 4 imágenes en validación
        k = min(samples_per_class, len(indices))
        val_indices.extend(rng.sample(indices, k))

    # DataLoader del Mini-Val
    transform = default_val_transform(cfg.img_size, tuple(cfg.normalize_mean), tuple(cfg.normalize_std))
    val_dataset = Subset(CLIPDataset(hf_val_ds, transform=transform), val_indices)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size * 2, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # --- Inicialización del Modelo ---
    model = SUPREME(cfg).to(device)
    trainable = get_trainable_params(model)

    bias_param_ids = {id(model.bpg.mu), id(model.bpg.sigma)}
    other_params = [p for p in trainable if id(p) not in bias_param_ids]
    param_groups = [
        {"params": other_params, "lr": cfg.lr},
        {"params": [model.bpg.mu, model.bpg.sigma], "lr": cfg.lr_bias},
    ]
    optimizer = torch.optim.SGD(param_groups, momentum=cfg.momentum)
    scaler = amp.GradScaler()

    # --- Bucle de Entrenamiento (Rápido, SIN Pruning) ---
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_one_epoch(model, loader, optimizer, scaler, class_names, device, cfg, epoch)
        
    # --- Evaluación Final (Paga el costo de prototipos una sola vez) ---
    model.eval()
    with torch.no_grad():
        txt_proto = compute_text_prototypes(model, class_names, device, ref_loader=loader)
        
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            img_emb = model.encode_image(images)
            
            sim = img_emb @ txt_proto.T
            preds = sim.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    final_val_acc = correct / total

    return final_val_acc

def run_optimization(config_path: str, n_trials: int = 30):
    """
    Lanza el estudio de optimización de hiperparámetros con Optuna.

    Crea (o reanuda) un estudio Optuna compartido mediante un JournalStorage en
    disco, lo que permite que múltiples nodos/workers contribuyan al mismo
    estudio en paralelo. Maximiza la exactitud de validación reportada por
    objective() durante n_trials trials.

    Parámetros
    ----------
    config_path : Ruta al archivo YAML de configuración base del experimento.
    n_trials    : Número de trials a ejecutar en este worker.
    """
    print(f"=== Iniciando Optimización Optuna (Fuerza Bruta Estratificada) ===")
    print(f"Buscando MAXIMIZAR 'Validation Accuracy' al final de la época 50.")
    
    # RUTA ABSOLUTA CRUCIAL para que múltiples nodos compartan el archivo
    log_path = "/home/svasquez/clip_prompt_learning_planktonzilla/optuna_supreme_val_bruteforce.log"
    storage = JournalStorage(JournalFileStorage(log_path))
    
    study = optuna.create_study(
        study_name="supreme_fewshot_val_bf", 
        storage=storage,
        load_if_exists=True,
        direction="maximize" 
        # Sin Pruner: evaluamos solo al final del trial
    )
    
    with tqdm(total=n_trials, desc="Trials de Optuna") as pbar:
        def callback(study, trial):
            pbar.update(1)
            tqdm.write(f"[Trial {trial.number}] Finalizado con Val Acc: {trial.value:.4f} | Mejor actual: {study.best_value:.4f}")

        study.optimize(lambda trial: objective(trial, config_path), n_trials=n_trials, callbacks=[callback])

    print("\n=== Optimización Completada ===")
    print(f"Mejor trial: {study.best_trial.number}")
    print(f"Mejor Val Acc alcanzado: {study.best_value:.4f}")
    print("Mejores hiperparámetros encontrados:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimizar hiperparámetros de SUPREME")
    parser.add_argument("--config", required=True, help="Ruta al archivo YAML base")
    parser.add_argument("--trials", type=int, default=10, help="Número de combinaciones a probar por worker")
    args = parser.parse_args()
    
    run_optimization(args.config, args.trials)