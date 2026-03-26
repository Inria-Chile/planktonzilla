"""
Funciones de puntuación OOD para el método SUPREME.

Convención: mayor puntuación = más dentro de la distribución (in-distribution).
Para usar como anomaly score, negar el valor retornado.

Notación:
  I       : (B, D)  embeddings de imagen normalizados
  I'      : (B, D)  f_img_txt(I) — imagen proyectada al espacio de texto
  P_txt   : (C, D)  prototipos de texto normalizados
  P_img   : (C, D)  prototipos de imagen normalizados
"""

import torch
import torch.nn.functional as F


def s_mcm(
    query: torch.Tensor,      # (B, D)
    prototypes: torch.Tensor, # (C, D) para Texto, o (C, K, D) para Imagen
    tau: float = 0.01,
) -> torch.Tensor:
    """
    Puntuación MCM (Maximum Concept Matching).

    Calcula max(softmax(sim / τ)) sobre los logits coseno entre la consulta
    y los prototipos de clase. Soporta dos modos según la forma de prototypes:

    - 2D (C, D): Caso estándar con prototipos promediados.
    - 3D (C, K, D): Caso multi-prototipo; usa la similitud del vecino más
      cercano por clase antes de aplicar softmax.

    Parámetros
    ----------
    query      : Embeddings de consulta normalizados, forma (B, D).
    prototypes : Prototipos de clase, forma (C, D) o (C, K, D).
    tau        : Temperatura de escala.

    Retorna
    -------
    Tensor de forma (B,) con las puntuaciones de confianza.
    """
    # 1. Caso Estándar (Prototipos de Texto promediados: 2D)
    if prototypes.dim() == 2:
        logits = (query @ prototypes.T) / tau  # (B, C)
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values
        
    # 2. Caso Multi-Prototipo (Imágenes K-Means intactas: 3D)
    elif prototypes.dim() == 3:
        # query: (B, D), prototypes: (C, K, D)
        # Calculamos la similitud coseno de cada imagen contra TODOS los K shots de TODAS las C clases
        sim = torch.einsum('bd,ckd->bck', query, prototypes) # Resultado: (B, C, K)
        
        # Para cada clase, nos quedamos con la similitud del shot más parecido (Nearest Neighbor)
        best_sim_per_class = sim.max(dim=-1).values  # (B, C)
        
        # Aplicamos la lógica MCM estándar sobre esas mejores similitudes
        logits = best_sim_per_class / tau
        probs = F.softmax(logits, dim=-1)
        
        return probs.max(dim=-1).values # (B,)


def s_mmp(
    img_emb: torch.Tensor,    # (B, D)
    txt_proto: torch.Tensor,  # (C, D)
    img_proto: torch.Tensor,  # (C, D)
    tau: float = 0.01,
) -> torch.Tensor:
    """
    Puntuación MMP (Multi-Modal Prototype).

    Promedio de dos puntuaciones MCM: una usando prototipos de texto y otra
    usando prototipos de imagen:
        S_MMP = (S_MCM(I, P_txt) + S_MCM(I, P_img)) / 2

    Parámetros
    ----------
    img_emb   : Embeddings de imagen normalizados, forma (B, D).
    txt_proto : Prototipos de texto normalizados, forma (C, D).
    img_proto : Prototipos de imagen normalizados, forma (C, D).
    tau       : Temperatura de escala.

    Retorna
    -------
    Tensor de forma (B,) con puntuaciones de confianza.
    """
    return (s_mcm(img_emb, txt_proto, tau) + s_mcm(img_emb, img_proto, tau)) / 2


def s_gmp(
    img_emb: torch.Tensor,    # (B, D)  I
    I_prime: torch.Tensor,    # (B, D)  f_img_txt(I)
    txt_proto: torch.Tensor,  # (C, D)
    img_proto: torch.Tensor,  # (C, D)
    tau: float = 0.01,
) -> torch.Tensor:
    """
    Puntuación GMP (Generalised Multi-modal Prototype).

    Promedio de cuatro puntuaciones MCM que combinan imagen original y
    proyectada con prototipos de texto e imagen:
        S_GMP = (S_MCM(I,  P_txt) + S_MCM(I,  P_img)
               + S_MCM(I', P_txt) + S_MCM(I', P_img)) / 4

    Parámetros
    ----------
    img_emb   : Embeddings de imagen normalizados, forma (B, D).
    I_prime   : Imagen proyectada al espacio de texto, forma (B, D).
    txt_proto : Prototipos de texto normalizados, forma (C, D).
    img_proto : Prototipos de imagen normalizados, forma (C, D).
    tau       : Temperatura de escala.

    Retorna
    -------
    Tensor de forma (B,) con puntuaciones de confianza.
    """
    s1 = s_mcm(img_emb, txt_proto, tau)
    s2 = s_mcm(img_emb, img_proto, tau)
    s3 = s_mcm(I_prime, txt_proto, tau)
    s4 = s_mcm(I_prime, img_proto, tau)
    return (s1 + s2 + s3 + s4) / 4
