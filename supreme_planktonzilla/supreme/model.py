"""
Arquitectura del modelo SUPREME.

Combina un backbone BioCLIP-2 congelado con dos módulos entrenables:
  - BPG (Biased Prompt Generation): genera prompts de texto condicionados
    a la imagen mediante tokens de contexto aprendibles y un sesgo gaussiano.
  - ITC (Image-Text Consistency): dos proyecciones lineales cruzadas que
    reducen la brecha entre modalidades imagen y texto.
"""

import torch
import torch.nn as nn
import open_clip

from .config import Config


class BiasedPromptGeneration(nn.Module):
    """
    Módulo BPG (Sección 3.2 del paper).

    Genera Image Domain-Biased Prompts (IDBP) que se alimentan al codificador
    de texto congelado para obtener prototipos de texto condicionados a la imagen.

    Componentes:
      - L tokens de contexto aprendibles V_i ∈ R^{N_lm}
      - MLP m(·): embedding de imagen → espacio de contexto
      - Sesgo de dominio gaussiano con media μ y factor de Cholesky σ
    """

    def __init__(self, cfg: Config):
        super().__init__()
        L, N = cfg.context_length, cfg.n_lm
        D = cfg.embed_dim

        # L tokens de contexto aprendibles V_i ∈ R^{N_lm}
        # Se inicializan con ceros; SUPREME los rellena desde una plantilla de texto.
        self.context_tokens = nn.Parameter(torch.zeros(L, N))

        # MLP m(·): embedding de imagen → espacio de contexto
        self.mlp = nn.Sequential(
            nn.Linear(D, N),
            nn.ReLU(),
            nn.Linear(N, N),
        )

        # Sesgo gaussiano de dominio: μ y factor triangular inferior de Cholesky σ
        self.mu = nn.Parameter(torch.zeros(N))
        # Se inicializa como identidad escalada para que la distribución inicial sea razonable
        self.sigma = nn.Parameter(torch.eye(N) * 0.01)

    def _sample_bias(self, training: bool) -> torch.Tensor:
        """
        Muestrea el sesgo de dominio.

        Durante entrenamiento retorna b = μ + σn con n ~ N(0, I).
        Durante evaluación retorna b = μ (valor esperado, sin ruido).

        Parámetros
        ----------
        training : Indica si el módulo está en modo entrenamiento.

        Retorna
        -------
        Tensor de forma (N,) con el sesgo muestreado o esperado.
        """
        if training:
            n = torch.randn_like(self.mu)
            L = torch.tril(self.sigma)
            return self.mu + L @ n
        return self.mu

    def forward(
        self,
        img_emb: torch.Tensor,      # (B, D)
        base_embs: torch.Tensor,    # (C, 77, N)  <- Ahora recibe la secuencia completa
        eos_indices: torch.Tensor,  # (C,)        <- Índices originales del token EOS
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Genera los prompts IDBP inyectando el contexto visual y el sesgo de dominio.

        Parámetros
        ----------
        img_emb     : Embeddings de imagen normalizados, forma (B, D).
        base_embs   : Secuencias de embeddings base del tokenizador CLIP, forma (C, 77, N).
        eos_indices : Índices del token EOS en cada secuencia de clase, forma (C,).

        Retorna
        -------
        tuple[Tensor, Tensor, Tensor]:
            idbp           : Secuencias IDBP ensambladas, forma (B, C, 77, N).
            b              : Sesgo de dominio muestreado expandido, forma (B, N).
            new_eos_indices: Índices EOS actualizados tras insertar L tokens de contexto, forma (C,).
        """
        B = img_emb.size(0)
        C, seq_len, N = base_embs.shape
        L = self.context_tokens.size(0)

        m_I = self.mlp(img_emb)                       # (B, N)
        b = self._sample_bias(self.training)          # (N,)
        b = b.unsqueeze(0).expand(B, -1)              # (B, N)

        # Generar los tokens de contexto: V_i + m(I) + b
        ctx = self.context_tokens.unsqueeze(0) + (m_I + b).unsqueeze(1)  # (B, L, N)
        ctx_exp = ctx.unsqueeze(1).expand(-1, C, -1, -1)                 # (B, C, L, N)

        # Extraer Prefix: Solo el token [SOS] (posición 0)
        prefix = base_embs[:, 0:1, :]  # (C, 1, N)
        prefix_exp = prefix.unsqueeze(0).expand(B, -1, -1, -1)           # (B, C, 1, N)

        # Extraer Suffix: Resto de la secuencia (clase, [EOS], padding)
        # Cortamos L posiciones al final para mantener la longitud original de 77
        suffix = base_embs[:, 1 : seq_len - L, :]  # (C, 76 - L, N)
        suffix_exp = suffix.unsqueeze(0).expand(B, -1, -1, -1)           # (B, C, 76 - L, N)

        # Ensamblar secuencia: [SOS] + Contexto + Clase + [EOS] + Padding
        idbp = torch.cat([prefix_exp, ctx_exp, suffix_exp], dim=2)       # (B, C, 77, N)

        # Actualizar la posición del token EOS (se desplazó L posiciones a la derecha)
        new_eos_indices = eos_indices + L

        return idbp, b, new_eos_indices


class ImageTextConsistency(nn.Module):
    """
    Módulo ITC (Sección 3.3 del paper).

    Dos proyecciones lineales cruzadas sin sesgo que reducen la brecha entre
    las modalidades de imagen y texto:
      f_img_txt: espacio de imagen → espacio de texto
      f_txt_img: espacio de texto  → espacio de imagen
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.embed_dim
        self.f_img_txt = nn.Linear(D, D, bias=False)
        self.f_txt_img = nn.Linear(D, D, bias=False)

    def forward(
        self,
        img_emb: torch.Tensor,   # (B, D)
        txt_emb: torch.Tensor,   # (C, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcula las proyecciones inter-modales y los ciclos de reconstrucción.

        Parámetros
        ----------
        img_emb : Embeddings de imagen normalizados, forma (B, D).
        txt_emb : Prototipos de texto normalizados, forma (C, D).

        Retorna
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]:
            I_prime     : f_img_txt(I) normalizado, forma (B, D).
            I_hat       : f_txt_img(f_img_txt(I)) normalizado, forma (B, D).
            P_hat       : f_img_txt(f_txt_img(P_txt)) normalizado, forma (C, D).
            P_img_space : f_txt_img(P_txt) normalizado, forma (C, D).
        """
        # Normalizar L2 las proyecciones inter-modales
        I_prime = torch.nn.functional.normalize(self.f_img_txt(img_emb), dim=-1)         # (B, D)
        P_img_space = torch.nn.functional.normalize(self.f_txt_img(txt_emb), dim=-1)     # (C, D)

        # Normalizar L2 los ciclos de reconstrucción
        I_hat = torch.nn.functional.normalize(self.f_txt_img(I_prime), dim=-1)           # (B, D)
        P_hat = torch.nn.functional.normalize(self.f_img_txt(P_img_space), dim=-1)       # (C, D)

        return I_prime, I_hat, P_hat, P_img_space


class SUPREME(nn.Module):
    """
    Modelo completo SUPREME.

    Contiene un backbone BioCLIP-2 congelado más los módulos entrenables
    BPG e ITC. El backbone nunca se actualiza durante el entrenamiento.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        print(f"Using pretrained backbone: {cfg.clip_pretrained}")
        # ── Backbone ──────────────────────────────────────────────────────────
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model, pretrained=cfg.clip_pretrained
        )
        self.clip = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(cfg.clip_model)

        # Congelar todos los parámetros del backbone CLIP
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # ── Módulos entrenables ───────────────────────────────────────────────
        self.bpg = BiasedPromptGeneration(cfg)
        self.itc = ImageTextConsistency(cfg)
        self.text_encode_chunk_size = cfg.text_encode_chunk_size

        # Inicializar context_tokens desde una plantilla de texto
        self._init_context_tokens(cfg.context_length)

    def get_preprocess(self):
        """Retorna el pipeline de preprocesado de imagen del backbone CLIP."""
        return self.preprocess
    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_context_tokens(self, L: int) -> None:
        """
        Inicializa los tokens de contexto desde los embeddings de una plantilla
        de texto. Los tokens de contexto se insertan como sufijo (después del
        token de clase), por lo que la plantilla debe representar texto que
        aparece naturalmente después del nombre de clase, e.g. "in the ocean".

        Si la plantilla tiene menos tokens que L, los tokens sobrantes se
        inicializan con ruido gaussiano pequeño.
        """
        template = "a photo of a"
        with torch.no_grad():
            tokens    = self.tokenizer([template])           # (1, 77)
            emb_table = self.clip.token_embedding
            embs      = emb_table(tokens.to(emb_table.weight.device)).squeeze(0).float()
            # embs: (77, N_lm)  —  posición 0 = SOS, 1..k = tokens de plantilla

            # Posiciones 1..k (tokens reales, sin SOS ni EOS)
            eot       = int(tokens[0].argmax().item())       # posición EOS
            word_embs = embs[1:eot]                          # (k, N_lm)
            k         = word_embs.size(0)

            init = torch.empty_like(self.bpg.context_tokens)
            if k >= L:
                init[:] = word_embs[:L]
            else:
                init[:k] = word_embs
                nn.init.normal_(init[k:], std=0.02)

        self.bpg.context_tokens.data.copy_(init)

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica imágenes y retorna embeddings normalizados.

        Parámetros
        ----------
        x : Lote de imágenes preprocesadas, forma (B, C, H, W).

        Retorna
        -------
        Tensor de forma (B, D) con embeddings de imagen normalizados (L2).
        """
        feats = self.clip.encode_image(x)
        return nn.functional.normalize(feats.float(), dim=-1)

    def encode_text_with_bpg(
        self,
        img_emb: torch.Tensor,
        class_names: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Genera prototipos de texto condicionados a la imagen mediante el BPG.

        Tokeniza las clases, genera los prompts IDBP y pasa las secuencias
        por el transformador de texto congelado para obtener los embeddings
        de clase normalizados.

        Parámetros
        ----------
        img_emb     : Embeddings de imagen normalizados, forma (B, D).
        class_names : Lista de nombres de clase (C elementos).

        Retorna
        -------
        tuple[Tensor, Tensor]:
            txt_proto : Prototipos de texto normalizados, forma (B, C, D).
            b         : Sesgo de dominio muestreado, forma (B, N).
        """
        device = img_emb.device

        # 1. Tokenizar clases (Devuelve forma: C, 77)
        tokens = self.tokenizer(class_names).to(device)

        # 2. Encontrar el token EOS (En CLIP, el EOS es el valor numérico máximo en la secuencia)
        eos_indices = tokens.argmax(dim=-1)  # (C,)

        # 3. Obtener los embeddings base de la tabla estática de CLIP
        emb_table = self.clip.token_embedding
        base_embs = emb_table(tokens)  # (C, 77, N_lm)

        # 4. Generar IDBP inyectando el contexto visual y sesgo
        idbp, b, new_eos_indices = self.bpg(img_emb, base_embs, eos_indices)

        B, C, seq_len, N = idbp.shape

        # Aplanar para pasar por el transformer congelado
        idbp_flat = idbp.view(B * C, seq_len, N)
        
        # Expandir los índices EOS para coincidir con la forma aplanada B*C
        eos_flat = new_eos_indices.unsqueeze(0).expand(B, -1).reshape(-1)

        # 5. Extraer características
        txt_feats = self._encode_text_from_embeddings(idbp_flat, eos_flat)  # (B*C, D)
        txt_proto = torch.nn.functional.normalize(txt_feats.float(), dim=-1).view(B, C, -1)

        return txt_proto, b

    def _encode_text_from_embeddings(self, token_embs: torch.Tensor, eos_indices: torch.Tensor) -> torch.Tensor:
        """
        Pasa secuencias de embeddings por el transformador de texto en chunks.

        Divide token_embs en fragmentos de tamaño text_encode_chunk_size para
        evitar problemas de memoria con lotes grandes (B*C secuencias).

        Parámetros
        ----------
        token_embs  : Secuencias de embeddings de entrada, forma (B*C, 77, N).
        eos_indices : Índices del token EOS por secuencia, forma (B*C,).

        Retorna
        -------
        Tensor de forma (B*C, D) con los embeddings de texto antes de normalizar.
        """
        # Se añaden los eos_indices al control de los chunks
        if len(token_embs) <= self.text_encode_chunk_size:
            return self._text_transformer_pass(token_embs, eos_indices)

        chunks = token_embs.split(self.text_encode_chunk_size)
        idx_chunks = eos_indices.split(self.text_encode_chunk_size)
        return torch.cat([self._text_transformer_pass(c, i) for c, i in zip(chunks, idx_chunks)], dim=0)

    def _text_transformer_pass(self, x: torch.Tensor, eos_indices: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta un chunk de secuencias por el transformador de texto de CLIP.

        Añade embeddings posicionales, aplica la máscara de atención causal,
        pasa por el transformador y extrae el vector del token EOS proyectado.
        Gestiona automáticamente el orden de dimensiones (batch-first vs seq-first)
        según el modelo cargado.

        Parámetros
        ----------
        x           : Chunk de embeddings de entrada, forma (N, T, D_lm).
        eos_indices : Índices del token EOS por secuencia del chunk, forma (N,).

        Retorna
        -------
        Tensor de forma (N, D) con los embeddings de texto proyectados.
        """
        N, T, _ = x.shape
        x = x + self.clip.positional_embedding[:T].unsqueeze(0)
        
        # Comprobamos dinámicamente cómo espera las dimensiones este modelo en particular
        is_batch_first = self.clip.transformer.resblocks[0].attn.batch_first
        
        # Solo permutamos si el modelo exige el formato antiguo (Seq, Batch, Dim)
        if not is_batch_first:
            x = x.permute(1, 0, 2)
            
        mask = self.clip.attn_mask
        if mask is not None:
            mask = mask[:T, :T].to(x.device)
            x = self.clip.transformer(x, attn_mask=mask)
        else:
            x = self.clip.transformer(x)
            
        # Devolvemos a la forma (Batch, Seq, Dim) si lo habíamos permutado
        if not is_batch_first:
            x = x.permute(1, 0, 2)
            
        x = self.clip.ln_final(x)
        
        # Extracción exacta del token EOS y proyección
        x = x[torch.arange(N, device=x.device), eos_indices]
        x = x @ self.clip.text_projection
        
        return x

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        class_names: list[str],
        labels: torch.Tensor,
    ) -> dict:
        """
        Forward pass completo utilizado durante el entrenamiento.

        Parámetros
        ----------
        images      : Lote de imágenes preprocesadas, forma (B, C, H, W).
        class_names : Lista de nombres de clase (C elementos).
        labels      : Etiquetas de clase del lote, forma (B,).

        Retorna
        -------
        dict con todos los tensores intermedios necesarios para calcular la pérdida:
          img_emb  : (B, D)   embeddings de imagen normalizados
          txt_proto: (C, D)   prototipos de texto
          I_prime  : (B, D)   f_img_txt(I)
          I_hat    : (B, D)   f_txt_img(f_img_txt(I))
          P_hat    : (C, D)   f_img_txt(f_txt_img(P_txt))
          P_img_sp : (C, D)   f_txt_img(P_txt)
          b        : (B, N)   sesgo gaussiano muestreado, expandido sobre el lote
          m_I      : (B, N)   salida del MLP para cada imagen
          labels   : (B,)     etiquetas de clase
        """
        B = images.size(0)

        # 1. Codificador de imagen congelado
        img_emb = self.encode_image(images)               # (B, D)

        # 2. BPG → prototipos de texto usando el embedding medio del lote.
        #    Reduce las llamadas al transformador de texto de B×C a solo C.
        img_emb_mean = img_emb.mean(0, keepdim=True)      # (1, D)
        txt_proto_mean, b_single = self.encode_text_with_bpg(img_emb_mean, class_names)
        # txt_proto_mean: (1, C, D),  b_single: (1, N)
        txt_proto = nn.functional.normalize(txt_proto_mean.squeeze(0), dim=-1)  # (C, D)
        b = b_single.expand(B, -1)                        # (B, N)

        # 3. Proyecciones ITC
        I_prime, I_hat, P_hat, P_img_sp = self.itc(img_emb, txt_proto)

        # 4. m(I) per-imagen para l_bias (salida bruta del MLP sobre cada imagen)
        m_I = self.bpg.mlp(img_emb)                       # (B, N)

        return dict(
            img_emb=img_emb,
            txt_proto=txt_proto,
            I_prime=I_prime,
            I_hat=I_hat,
            P_hat=P_hat,
            P_img_sp=P_img_sp,
            b=b,
            m_I=m_I,
            labels=labels,
        )
