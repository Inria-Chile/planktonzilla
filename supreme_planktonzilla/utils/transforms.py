"""
Transformaciones de imagen estándar compatibles con los modelos CLIP.

Utiliza la normalización de ImageNet, que es la convención empleada por la
mayoría de los backbones CLIP (incluyendo BioCLIP-2 y OpenCLIP).
"""

from torchvision import transforms


_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)   


def default_train_transform(
    image_size: int = 224,
    mean: tuple = _CLIP_MEAN,
    std:  tuple = _CLIP_STD,
) -> transforms.Compose:
    """
    Transformación de entrenamiento con aumentación de datos.

    Aplica un recorte aleatorio con redimensionado, volteo horizontal aleatorio
    y normalización estándar de CLIP.

    Parámetros
    ----------
    image_size : Tamaño del lado de la imagen cuadrada de salida (píxeles).
    mean       : Media de normalización por canal (R, G, B).
    std        : Desviación estándar de normalización por canal (R, G, B).

    Retorna
    -------
    transforms.Compose : Pipeline de transformación para entrenamiento.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def default_val_transform(
    image_size: int = 224,
    mean: tuple = _CLIP_MEAN,
    std:  tuple = _CLIP_STD,
) -> transforms.Compose:
    """
    Transformación de validación/evaluación sin aumentación de datos.

    Aplica redimensionado, recorte central y normalización estándar de CLIP.

    Parámetros
    ----------
    image_size : Tamaño del lado de la imagen cuadrada de salida (píxeles).
    mean       : Media de normalización por canal (R, G, B).
    std        : Desviación estándar de normalización por canal (R, G, B).

    Retorna
    -------
    transforms.Compose : Pipeline de transformación para evaluación.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
