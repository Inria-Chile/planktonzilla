"""
Configuration-based experiment runner for OOD detection.

This script reads experiment parameters from a YAML configuration file
and runs OOD detection experiments in three composable stages:

  --fit              Fit OOD methods on training data and save parameters
  --predict          Run inference on ID/OOD datasets and save scores
  --generate_results Compute metrics from saved scores

Usage:
    python run_from_config.py --config planktonzilla_definitive --fit --predict --generate_results
"""

import argparse
import gc
import torch

from ood_detection.core import EngineOOD
from ood_detection.datasets import load_dataset_from_config
from ood_detection.utils import (
    set_seed,
    get_preprocessing,
    ExperimentLogger,
    resolve_config_path,
    load_config,
    parse_ood_method_config,
)


def main():
    """Main experiment runner from config."""
    parser = argparse.ArgumentParser(description="Run OOD detection from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="example_config",
        help="Name of YAML config file in configs/ directory (without .yaml extension)"
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit OOD methods on the training dataset and save fitted parameters."
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run inference on ID/OOD datasets and save raw scores."
    )
    parser.add_argument(
        "--generate_results",
        action="store_true",
        help="Compute OOD metrics from saved scores."
    )
    args = parser.parse_args()

    # Initialize logger
    logger = ExperimentLogger()

    # Validate that at least one stage is requested
    if not (args.fit or args.predict or args.generate_results):
        logger.error("At least one of --fit, --predict, --generate_results must be specified.")
        return

    logger.info(f"Config name provided: {args.config}")
    logger.info(f"Stages: fit={args.fit}, predict={args.predict}, generate_results={args.generate_results}")
     
    # Resolve config path
    try:
        config_path = resolve_config_path(args.config)
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"{e}")
        return

    logger.start_timer("total_experiment")

    # Extract configuration sections
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    dataloader_cfg = config["dataloader"]
    ood_methods_cfg = config["ood_methods"]
    experiment_cfg = config["experiment"]
    advanced_cfg = config.get("advanced", {})
    
    # Set device
    device = model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    num_labels = model_cfg.get("num_id_classes")

    # Set random seed
    seed = experiment_cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    experiment_name = experiment_cfg["name"]

    save_fitted_params = experiment_cfg.get("save_fitted_params", None)
    load_fitted_params = experiment_cfg.get("load_fitted_params", None)

    # Validate config requirements per flag
    if args.fit:
        assert save_fitted_params is not None, \
            "save_fitted_params must be specified in config when using --fit"
    
    if args.predict and not args.fit:
        assert load_fitted_params is not None, \
            "load_fitted_params must be specified in config when using --predict without --fit"

    # --- Load datasets based on requested stages ---
    ds_fit, ds_id, ds_ood = None, None, None

    if args.fit:
        fit_cfg = dataset_cfg["fit"]
        ds_fit = load_dataset_from_config(
            source=fit_cfg["source"],
            shards = fit_cfg.get("shards",None),
            split=fit_cfg.get("split", "train"),
            classes=fit_cfg.get("classes"),
            samples_per_class=fit_cfg.get("samples_per_class"),
            logger=logger
        )
        logger.info(f"Fit dataset loaded: {len(ds_fit)} samples, {num_labels} classes")
    
    if args.predict:
        id_cfg = dataset_cfg["id"]
        if id_cfg["source"] is not None:
            ds_id = load_dataset_from_config(
                source=id_cfg["source"],
                shards = id_cfg.get("shards",None),
                split=id_cfg.get("split", "test"),
                classes=id_cfg.get("classes"),
                samples_per_class=id_cfg.get("samples_per_class"),
                logger=logger
            )
            logger.info(f"ID dataset loaded: {len(ds_id)} samples")

        ood_cfg = dataset_cfg["ood"]
        if ood_cfg["source"] is not None:
            ds_ood = load_dataset_from_config(
                source=ood_cfg["source"],
                shards = ood_cfg.get("shards",None),
                split=ood_cfg.get("split", "train"),
                classes=ood_cfg.get("classes"),
                samples_per_class=ood_cfg.get("samples_per_class"),
                logger=logger
            )
            logger.info(f"OOD dataset loaded: {len(ds_ood)} samples")

    # --- Engine setup ---
    #TODO: cambiar backbone por classifier (va a ser un clasificador entrenado con el número de clases ya definido)
    backbone = model_cfg["backbone"]
    logger.info(f"Backbone: {backbone}")

    gpu_count = torch.cuda.device_count() if device == "cuda" else 1
    batch_size = dataloader_cfg.get("batch_size", 64)
    if advanced_cfg.get("auto_scale_batch_size", True) and gpu_count > 1:
        batch_size = batch_size * gpu_count
        logger.info(f"Auto-scaled batch size to {batch_size} ({gpu_count} GPUs)")
    
    save_results = experiment_cfg.get("save_results", False)
    # If --predict is requested, we always need to save scores for --generate_results
    if args.predict:
        save_results = True

    engine = EngineOOD(
        backbone=backbone,
        n_id_classes=num_labels,
        device=device,
        experiment_name=experiment_name,
        save_results=save_results,
        save_fitted_params=save_fitted_params,
        load_fitted_params=load_fitted_params,
        logger=logger,
    )
    logger.info("Engine initialized successfully.")

    # Set up dataloaders
    logger.start_timer("dataloader_setup")
    engine.set_dataloaders(
        id_ds=ds_id,
        ood_ds=ds_ood,
        fit_ds=ds_fit,
        batch_size=batch_size,
        preprocessing=get_preprocessing(backbone),
        num_workers=dataloader_cfg.get("num_workers", 4)
    )
    logger.end_timer("dataloader_setup")
    logger.info("Dataloaders ready.")

    # Parse OOD method configs
    methods = [parse_ood_method_config(m) for m in ood_methods_cfg]

    # ==================== STAGE 1: FIT ====================
    if args.fit:
        logger.info("=" * 60)
        logger.info("STAGE: FIT")
        logger.info("=" * 60)
        engine.fit_methods(method=methods)
        logger.info("Fitted parameters saved.")

    # ==================== STAGE 2: PREDICT ====================
    if args.predict:
        logger.info("=" * 60)
        logger.info("STAGE: PREDICT")
        logger.info("=" * 60)

        # If we just fitted, methods are already loaded; otherwise load from disk
        if not args.fit:
            engine.load_methods(method=methods)
        
        engine.predict()
        logger.info(f"Scores saved to: {engine.results_path}")

    # ==================== STAGE 3: GENERATE RESULTS ====================
    if args.generate_results:
        logger.info("=" * 60)
        logger.info("STAGE: GENERATE RESULTS")
        logger.info("=" * 60)

        # We need methods registered to know which score files to load
        if not args.predict and not args.fit:
            # Register methods without fitting (just need the names)
            engine.load_methods(method=methods)
        
        results = engine.compute_results()

        # Save aggregated results
        results_file = f"{engine.results_path}/ood_results.pt"
        torch.save(results, results_file)
        logger.info(f"Results saved to: {results_file}")
        
        # Print results
        logger.info("=" * 60)
        logger.info(f"Results for {backbone}")
        logger.info("=" * 60)
        for method_name, metrics in results.items():
            logger.info(f"Method: {method_name}")
            if metrics is not None:
                logger.info(str(metrics))
            else:
                logger.error("  Method failed")
            logger.info("-" * 60)

    # Cleanup
    logger.info("Cleaning up...")
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.end_timer("total_experiment")
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
