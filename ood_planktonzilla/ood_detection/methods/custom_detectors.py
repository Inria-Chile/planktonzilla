from pytorch_ood.detector import ViM, KLMatching, Mahalanobis,EnergyBased, MaxLogit, ReAct, MaxSoftmax
import torch.nn as nn
import datetime 
from typing import Callable, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_ood.utils import TensorBuffer, is_known
#TODO: Para cada clase guardar en el mismo directorio la descripcion del experimento (config, modelo, etc) junto con los parametros ajustados para cada metodo. Esto facilita la trazabilidad y reproducibilidad de los experimentos.

def custom_extract_features(
    data_loader: DataLoader, 
    model: Callable[[Tensor], Tensor], 
    device: Optional[str]
) -> Tuple[Tensor, Tensor]:
    """
    Helper to extract outputs from model with a progress bar. Ignores OOD inputs.
    ref: https://github.com/kkirchheim/pytorch-ood/blob/dev/src/pytorch_ood/utils/utils.py#L322
    """
    buffer = TensorBuffer()

    with torch.no_grad():

        for batch in tqdm(data_loader, desc="Extracting features", leave=True):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            known = is_known(y)
            if known.any():
                z = model(x[known])
                z = z.view(known.sum(), -1)  # flatten
                buffer.append("embedding", z)
                buffer.append("label", y[known])

        if buffer.is_empty():
            raise ValueError("No ID instances in loader")

    z = buffer.get("embedding")
    y = buffer.get("label")
    return z, y

class CustomEnergyBased(EnergyBased):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomMaxLogit(MaxLogit):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomReAct(ReAct):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomMaxSoftmax(MaxSoftmax):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomKLMatching(KLMatching):
    def fit(self, data_loader: DataLoader, device="cpu"):
        """
        Estimates typical distributions for each class.
        Ignores OOD samples.

        :param data_loader: validation data loader
        :param device: device which should be used for calculations
        ref: https://github.com/kkirchheim/pytorch-ood/blob/dev/src/pytorch_ood/detector/klmatching.py
        """
        if self.model is None:
            raise ModelNotSetException

        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)

        logits, labels = custom_extract_features(data_loader, self.model, device)
        return self.fit_features(logits, labels, device)

    def save_fitted_parameters(self,save_path:str):
        #Save fitted parameters for later use in evaluation 
        assert len(self.dists) !=0, "No fitted parameters to save. Please fit the detector first."
        torch.save(self.dists.state_dict(), f"{save_path}/kl_matching_parameters.pt")
    
    def load_fitted_parameters(self,load_path:str): 
        state_dict = torch.load(f"{load_path}/kl_matching_parameters.pt",map_location="cpu") 
        self.dists = nn.ParameterDict()
        for key, tensor in state_dict.items():
            self.dists[key] = nn.Parameter(tensor)

class CustomMahalanobis(Mahalanobis):
    def fit(self, data_loader: DataLoader, device: str = None):
        """
        Fit parameters of the multi variate gaussian.

        :param data_loader: dataset to fit on.
        :param device: device to use
        """
        if device is None:
            device = list(self.model.parameters())[0].device


        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)

        z, y = custom_extract_features(data_loader, self.model, device)
        return self.fit_features(z, y, device)

    def save_fitted_parameters(self,save_path:str): #Save fitted parameters
        #Save fitted parameters for later use in evaluation
        if self.mu is not None and self.precision is not None:
            d = {"mu":self.mu,
                "precision":self.precision,
                }
            torch.save(d, f"{save_path}/mahalanobis_parameters.pt") 
    def load_fitted_parameters(self,load_path:str): 
        params = torch.load(f"{load_path}/mahalanobis_parameters.pt") 
        self.mu = params["mu"] 
        self.precision = params["precision"]

class CustomViM(ViM):
    def fit(self, data_loader, device="cpu"):
        """
        Extracts features and logits, computes principle subspace and alpha. Ignores OOD samples.

        :param data_loader: dataset to fit on
        :param device: device to use
        :return:
        """
        try:
            from sklearn.covariance import EmpiricalCovariance
        except ImportError:
            raise Exception("You need to install sklearn to use ViM.")

        if self.model is None:
            raise ModelNotSetException

        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)

        features, labels = custom_extract_features(data_loader, self.model, device)
        return self.fit_features(features, labels)

    def save_fitted_parameters(self,save_path:str):
        #Save fitted parameters for later use in evaluation
        if self.alpha is not None and self.principal_subspace is not None:

            d = {
            "alpha":self.alpha,
            "principal_subspace":self.principal_subspace,
            }

            torch.save(d, f"{save_path}/vim_parameters.pt")

    def load_fitted_parameters(self,load_path:str): 
        params = torch.load(f"{load_path}/vim_parameters.pt",map_location="cpu", weights_only=False)
        self.alpha = params["alpha"] 
        self.principal_subspace = params["principal_subspace"]

