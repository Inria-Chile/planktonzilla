"""
Funciones de pérdida del método SUPREME.

Todas las pérdidas operan sobre embeddings normalizados.

Notación (consistente con el paper):
  I         : (B, D)  embeddings de imagen normalizados
  P_txt     : (C, D)  prototipos de texto normalizados (uno por clase)
  I'        : (B, D)  f_img_txt(I)
  I_hat     : (B, D)  f_txt_img(f_img_txt(I))
  P_hat     : (C, D)  f_img_txt(f_txt_img(P_txt))
  P_img_sp  : (C, D)  f_txt_img(P_txt)
  b         : (B, N)  sesgo gaussiano muestreado
  m_I       : (B, N)  salida del MLP para cada imagen
"""

import torch
import torch.nn.functional as F
from .config import Config


def cosine_logits(
    a: torch.Tensor, b: torch.Tensor, tau: float
) -> torch.Tensor:
    """
    Calcula logits de similitud coseno escalados por 1/τ.

    Parámetros
    ----------
    a   : Tensor de forma (B, D).
    b   : Tensor de forma (C, D).
    tau : Temperatura de escala.

    Retorna
    -------
    Tensor de forma (B, C) con los logits.
    """
    return (a @ b.T) / tau


def l_id(
    img_emb: torch.Tensor,   # (B, D)
    txt_proto: torch.Tensor, # (C, D)
    labels: torch.Tensor,    # (B,)  long
    tau: float,
) -> torch.Tensor:
    """
    Pérdida de clasificación ID por entropía cruzada.

    Mide qué tan bien los embeddings de imagen se alinean con los prototipos
    de texto de su clase correcta.

    Parámetros
    ----------
    img_emb   : Embeddings de imagen normalizados, forma (B, D).
    txt_proto : Prototipos de texto normalizados, forma (C, D).
    labels    : Etiquetas de clase, forma (B,).
    tau       : Temperatura CLIP.

    Retorna
    -------
    Tensor escalar con el valor de la pérdida.
    """
    logits = cosine_logits(img_emb, txt_proto, tau)  # (B, C)
    return F.cross_entropy(logits, labels)


def l_inter(
    img_emb: torch.Tensor,    # (B, D)  I
    txt_proto: torch.Tensor,  # (C, D)  P_txt
    I_prime: torch.Tensor,    # (B, D)  f_img_txt(I)
    P_img_sp: torch.Tensor,   # (C, D)  f_txt_img(P_txt)
    labels: torch.Tensor,     # (B,)
    tau: float,
) -> torch.Tensor:
    """
    Pérdida de consistencia inter-modal.

    Fomenta que I se alinee con f_txt_img(P) y que f_img_txt(I) se alinee
    con P_txt, reduciendo la brecha entre modalidades.

    Parámetros
    ----------
    img_emb  : Embeddings de imagen normalizados, forma (B, D).
    txt_proto: Prototipos de texto normalizados, forma (C, D).
    I_prime  : Imagen proyectada al espacio de texto, forma (B, D).
    P_img_sp : Prototipos de texto proyectados al espacio de imagen, forma (C, D).
    labels   : Etiquetas de clase, forma (B,).
    tau      : Temperatura CLIP.

    Retorna
    -------
    Tensor escalar con el valor de la pérdida.
    """
    logits_a = cosine_logits(img_emb, P_img_sp, tau)   # (B, C)
    logits_b = cosine_logits(I_prime, txt_proto, tau)   # (B, C)
    loss = (
        F.cross_entropy(logits_a, labels)
        + F.cross_entropy(logits_b, labels)
    )
    return loss


def l_intra(
    img_emb: torch.Tensor,    # (B, D)  I
    txt_proto: torch.Tensor,  # (C, D)  P_txt
    I_hat: torch.Tensor,      # (B, D)  f_txt_img(f_img_txt(I))
    P_hat: torch.Tensor,      # (C, D)  f_img_txt(f_txt_img(P_txt))
) -> torch.Tensor:
    """
    Pérdida de consistencia intra-modal por ciclo (L1).

    Penaliza el error de reconstrucción en ida y vuelta dentro de cada
    modalidad: imagen → texto → imagen y texto → imagen → texto.

    Parámetros
    ----------
    img_emb  : Embeddings de imagen originales, forma (B, D).
    txt_proto: Prototipos de texto originales, forma (C, D).
    I_hat    : Imagen reconstruida tras el ciclo, forma (B, D).
    P_hat    : Prototipo de texto reconstruido tras el ciclo, forma (C, D).

    Retorna
    -------
    Tensor escalar con el valor de la pérdida.
    """
    loss_img = (img_emb - I_hat).abs().sum(dim=-1).mean()
    loss_txt = (txt_proto - P_hat).abs().sum(dim=-1).mean()
    return loss_img + loss_txt


def l_bias(
    mu: torch.Tensor,   # (N,)  media aprendible
    b: torch.Tensor,    # (B, N) sesgo muestreado
    m_I: torch.Tensor,  # (B, N) salida del MLP
) -> torch.Tensor:
    """
    Pérdida de regularización del sesgo de dominio.

    Acerca tanto μ como el sesgo muestreado b hacia la proyección m(I) del
    MLP, anclando el sesgo gaussiano a la distribución de imágenes ID.

    Parámetros
    ----------
    mu  : Media aprendible del sesgo gaussiano, forma (N,).
    b   : Sesgo muestreado en el forward pass, forma (B, N).
    m_I : Salida del MLP para cada imagen, forma (B, N).

    Retorna
    -------
    Tensor escalar con el valor de la pérdida.
    """
    mu_exp = mu.unsqueeze(0).expand_as(m_I)           # (B, N)
    loss = (
        (mu_exp - m_I).abs().sum(dim=-1).mean()
        + (b - m_I).abs().sum(dim=-1).mean()
    )
    return loss


def total_loss(
    fwd: dict,
    cfg: "Config",
    mu: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Calcula la pérdida combinada de SUPREME a partir del diccionario de forward.

    Pérdida total = l_id + α·(l_inter + l_intra) + β·l_bias

    Parámetros
    ----------
    fwd : Diccionario de tensores retornado por SUPREME.forward().
    cfg : Objeto de configuración con los pesos de pérdida (alpha, beta, tau).
    mu  : Media aprendible del módulo BPG, forma (N,).

    Retorna
    -------
    tuple[Tensor, dict] : Pérdida total escalar y diccionario con los
                          componentes individuales para registro (logging).
    """
    tau = cfg.tau
    labels = fwd["labels"]

    loss_id = l_id(fwd["img_emb"], fwd["txt_proto"], labels, tau)
    loss_inter = l_inter(
        fwd["img_emb"], fwd["txt_proto"],
        fwd["I_prime"], fwd["P_img_sp"],
        labels, tau,
    )
    loss_intra = l_intra(
        fwd["img_emb"], fwd["txt_proto"],
        fwd["I_hat"], fwd["P_hat"],
    )
    loss_b = l_bias(mu, fwd["b"], fwd["m_I"])

    total = (
        loss_id
        + cfg.alpha * (loss_intra + loss_inter)
        + cfg.beta * loss_b
    )

    components = dict(
        loss_id=loss_id.item(),
        loss_inter=loss_inter.item(),
        loss_intra=loss_intra.item(),
        loss_bias=loss_b.item(),
        loss_total=total.item(),
    )
    return total, components
