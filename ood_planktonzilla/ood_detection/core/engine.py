"""Main OOD detection engine for orchestrating experiments."""

import os
import datetime
from typing import List, Union, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_ood.utils import OODMetrics

from ..models import load_model, get_features_extractor, get_head_layer, TorchModel
from ..datasets import HFImageDataset, OODLabelTransform
from ..methods import fit_detection_method, load_detection_method, apply_ood_method


class EngineOOD:
    """
    Main engine for OOD detection experiments.
    
    This class orchestrates the entire OOD detection pipeline, including:
    - Model loading and initialization
    - Feature extractor creation
    - DataLoader management
    - OOD method configuration (fit or load)
    - Prediction (inference scores)
    - Results computation and saving
    - Multi-GPU parallelization
    
    Args:
        backbone (str): Name of the model backbone to use.
        n_id_classes (int): Number of in-distribution classes.
        save_results (bool): Whether to save results to disk. Default is False.
        device (str): Device to use ('cpu' or 'cuda'). Default is 'cpu'.
        save_fitted_params (str, optional): Directory name to save fitted parameters.
        load_fitted_params (str, optional): Directory name to load fitted parameters.
        logger (ExperimentLogger): Logger for timing and messages.
    """
    
    def __init__(self, backbone: str, n_id_classes: int, 
                 save_results: Optional[bool] = False,
                 device: str = "cpu",
                 experiment_name:str="",
                 save_fitted_params: Optional[str] = None,
                 load_fitted_params: Optional[str] = None,
                 logger=None):
        self.logger = logger
        self.t = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.experiment_name=experiment_name
        self.save_results = save_results
        self.save_fitted_params = os.path.join("methods_params",save_fitted_params) if save_fitted_params else None
        self.load_fitted_params = os.path.join("methods_params",load_fitted_params) if load_fitted_params else None

        # Create results directory if saving
        os.makedirs("ood_results", exist_ok=True)
        os.makedirs(f"ood_results/{experiment_name}_{self.t}")
        if self.save_fitted_params:
            os.makedirs(self.save_fitted_params, exist_ok=True)
        self.results_path = f"ood_results/{experiment_name}_{self.t}"
        
        # Set device
        try:
            self.device = torch.device(device)
        except:
            self.device = torch.device("cpu")
        
        self.backbone_name = backbone
        
        # 1. Load Model
        self.base_model = load_model(backbone=self.backbone_name, n_classes=n_id_classes)
        
        # 2. Wrap in TorchModel
        self.base_model = TorchModel(self.base_model)
        self.base_model.to(self.device)
        
        # 3. Create Feature Extractor
        self.feature_extractor = get_features_extractor(backbone=backbone, model=self.base_model)
        self.feature_extractor.to(self.device)
        
        # 4. Get Head Layer
        self.head_layer = get_head_layer(backbone, model=self.base_model)
        
        # --- PARALLELIZATION BLOCK ---
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Parallelizing on {torch.cuda.device_count()} GPUs!")
            self.base_model = nn.DataParallel(self.base_model)
            self.feature_extractor = nn.DataParallel(self.feature_extractor)
        # -----------------------------
        
        self.detector = OODMetrics()
        self.methods = {}
        self._models_to_eval()
    
    def _models_to_eval(self):
        """Set all models to evaluation mode."""
        self.base_model.eval()
        self.feature_extractor.eval()
        if self.head_layer is not None:
            self.head_layer.eval()
    
    
    def set_dataloaders(self, id_ds, ood_ds, fit_ds, preprocessing, 
                       batch_size=32, num_workers=0):
        """
        Create Torch DataLoaders for ID, OOD, and fitting datasets.
        
        Args:
            id_ds: In-distribution dataset (HuggingFace dataset).
            ood_ds: Out-of-distribution dataset (HuggingFace dataset).
            fit_ds: Dataset for fitting OOD detectors (HuggingFace dataset).
            preprocessing: Torchvision transforms to apply to images.
            batch_size (int): Batch size for DataLoaders. Default is 32.
            num_workers (int): Number of worker processes for data loading. Default is 0.
        """
        # Increased num_workers helps keep GPUs fed
        self.fit_dataloader, self.ood_dataloader, self.id_dataloader = None, None, None
        if fit_ds is not None:
            self.fit_dataloader = DataLoader(
                HFImageDataset(fit_ds, preprocessing),
                batch_size=batch_size, 
                num_workers=num_workers
            )

        if ood_ds is not None:
            self.ood_dataloader = DataLoader(
                HFImageDataset(ood_ds, preprocessing, OODLabelTransform(-1)),
                batch_size=batch_size, 
                num_workers=num_workers
            )

        if id_ds is not None:
            self.id_dataloader = DataLoader(
                HFImageDataset(id_ds, preprocessing),
                batch_size=batch_size, 
                num_workers=num_workers
            )
    
    def _prepare_method_config(self, m: dict):
        """
        Select the correct model and enrich the method config with engine-specific params.
        
        Args:
            m (dict): Method configuration with 'name' and 'cfg' keys.
            
        Returns:
            tuple: (model_to_use, enriched_method_config)
        """
        method_name = m["name"]
        
        # Select Model
        if method_name in ["MSP", "KL-Matching", "Energy", "MaxLogit"]:
            model_to_use = self.base_model
        elif method_name in ["Mahalanobis", "ViM", "React"]:
            model_to_use = self.feature_extractor
        else:
            model_to_use = self.base_model
        
        # Configure specific parameters
        if method_name == "React":
            m["cfg"]["head"] = self.head_layer
        
        if method_name == "ViM":
            layer = self.head_layer
            if isinstance(layer, nn.DataParallel):
                layer = layer.module
            m["cfg"]["w"] = layer.weight
            m["cfg"]["b"] = layer.bias
        
        return model_to_use, m
    
    def fit_methods(self, method: Union[List[dict], dict]):
        """
        Fit OOD detection methods on the fit dataset and save parameters.
        
        Args:
            method: Single method config dict or list of dicts.
        """
        methods_list = [method] if isinstance(method, dict) else method
        
        for m in methods_list:
            method_name = m["name"]
            model_to_use, m = self._prepare_method_config(m)
            
            self.logger.info(f"Fitting {method_name}...")
            self.logger.start_timer(f"fitting_{method_name}")
            
            ood_method = fit_detection_method(
                model=model_to_use,
                dict_method=m,
                fit_dataloader=self.fit_dataloader,
                device=self.device,
                save_fitted_params=self.save_fitted_params,
            )
            
            self.logger.end_timer(f"fitting_{method_name}")
            self.methods[method_name] = ood_method
    
    def load_methods(self, method: Union[List[dict], dict]):
        """
        Load pre-fitted OOD detection methods from saved parameters.
        
        Args:
            method: Single method config dict or list of dicts.
        """
        methods_list = [method] if isinstance(method, dict) else method
        
        for m in methods_list:
            method_name = m["name"]
            model_to_use, m = self._prepare_method_config(m)
            
            self.logger.info(f"Loading parameters for {method_name}...")
            self.logger.start_timer(f"loading_{method_name}")
            
            ood_method = load_detection_method(
                model=model_to_use,
                dict_method=m,
                load_fitted_params=self.load_fitted_params,
            )
            
            self.logger.end_timer(f"loading_{method_name}")
            self.methods[method_name] = ood_method
    
    def predict(self):
        """
        Run inference on ID and OOD datasets and collect scores.
        
        Saves raw scores per method if ``self.save_results`` is True.
        Does **not** compute metrics — use ``compute_results()`` for that.
        """
        for method_name, method in self.methods.items():
            self.logger.start_timer(f"predicting_{method_name}")
            try:
                scores_ood, labels_ood = apply_ood_method(
                    self.ood_dataloader, method, self.device, method_name
                )
                self.detector.update(scores_ood, labels_ood)
                
                scores_id, labels_id = apply_ood_method(
                    self.id_dataloader, method, self.device, method_name
                )
                self.detector.update(scores_id, labels_id)
            except Exception as e:
                self.logger.error(f"Error predicting with {method_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if self.save_results:
                    self.detector.buffer.save(
                        path=f"{self.results_path}/ood_results_{method_name}.pt"
                    )
                self.detector.reset()
            self.logger.end_timer(f"predicting_{method_name}")
    
    def compute_results(self):
        """
        Compute OOD metrics from saved score files.
        
        Loads the score buffers saved by ``predict()`` and computes metrics.
            
        Returns:
            dict: Dictionary mapping method names to their computed metrics.
        """
        from pytorch_ood.utils import OODMetrics
        results = {}
        
        for method_name in self.methods:
            self.logger.start_timer(f"computing_{method_name}")
            try:
                score_path = f"{self.results_path}/ood_results_{method_name}.pt"
                detector = OODMetrics()
                detector.buffer.load(path=score_path)
                results[method_name] = detector.compute()
            except Exception as e:
                self.logger.error(f"Error computing results for {method_name}: {e}")
                import traceback
                traceback.print_exc()
                results[method_name] = None
            self.logger.end_timer(f"computing_{method_name}")
        
        return results
