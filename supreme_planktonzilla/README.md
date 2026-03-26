# suPreMe Framework

Framework de detección OOD basado en suPreMe, un método que opera sobre embeddings de CLIP.

---

## Módulos de `supreme/`

### `supreme/__init__.py`
Punto de entrada público del paquete. Exporta las funciones `run_train` y `run_evaluate` para que `main.py` pueda invocar entrenamiento y evaluación.

---

### `supreme/config.py`
Define el dataclass `Config` con todos los hiperparámetros y la configuración de datasets necesaria para entrenamiento y evaluación. Soporta carga desde archivo YAML mediante `Config.from_yaml(path)`. Cada dataset se define con un `DatasetConfig` que especifica `source`, `split` e `idx_file`.

---

### `supreme/model.py`
Implementa la arquitectura completa del modelo SUPREME. Combina un backbone tipo CLIP **congelado** con dos módulos entrenables:

- **BPG (Biased Prompt Generation):** genera prompts de texto condicionados a la imagen mediante `L` tokens de contexto aprendibles y un sesgo de dominio gaussiano (μ, σ).
- **ITC (Image-Text Consistency):** dos proyecciones lineales cruzadas (`f_img_txt`, `f_txt_img`) que reducen la brecha entre modalidades imagen y texto.

---

### `supreme/data.py`
Constructores de `DataLoader` para el método SUPREME. Orquesta la carga de datasets HuggingFace, la construcción del wrapper `CLIPDataset` y la creación de los DataLoaders para el conjunto de soporte (few-shot) y los conjuntos de evaluación.

---

### `supreme/losses.py`
Define las funciones de pérdida del método SUPREME. Todas operan sobre embeddings normalizados.

| Función | Descripción |
|---|---|
| `l_id` | Pérdida de clasificación ID por entropía cruzada entre imagen y prototipos de texto |
| `l_inter` | Pérdida de consistencia inter-modal: alinea imagen↔texto en ambas direcciones |
| `l_intra` | Pérdida de ciclo intra-modal (L1): penaliza el error de reconstrucción imagen→texto→imagen y texto→imagen→texto |
| `l_bias` | Regularización del sesgo de dominio: acerca μ y b hacia la proyección m(I) del MLP |
| `total_loss` | Combina todas: `l_id + α·(l_inter + l_intra) + β·l_bias` |

---

### `supreme/prototypes.py`
Calcula los prototipos de imagen y texto para inferencia OOD.

- **Prototipos de imagen:** tensor `(C, K, D)` con los K embeddings por clase (multi-prototipo, sin promediar) del conjunto de soporte.
- **Prototipos de texto:** usa el BPG entrenado en modo inferencia (`b = μ`, sin ruido) para generar embeddings de texto condicionados a la imagen.

---

### `supreme/scores.py`
Funciones de puntuación OOD. Convención: **mayor puntuación = más in-distribution**.

| Función | Descripción |
|---|---|
| `s_mcm` | Maximum Concept Matching: `max(softmax(sim / τ))`. Soporta prototipos 2D `(C,D)` y 3D `(C,K,D)` (k-NN sobre shots) |
| `s_mmp` | Multi-Modal Prototype: promedio de dos MCM, uno con prototipos de texto y otro de imagen |
| `s_gmp` | Generalised Multi-modal Prototype: promedio de cuatro MCM combinando imagen original, imagen proyectada y ambos tipos de prototipos |

---

### `supreme/train.py`
Bucle de entrenamiento de SUPREME. Ejecuta `cfg.trials` ejecuciones independientes con semillas distintas, cada una con `cfg.epochs` épocas usando SGD con precisión mixta (AMP). Al finalizar cada trial calcula y cachea los prototipos de imagen y texto dentro del checkpoint.

**Archivos generados:**

```
models/
└── supreme/
    └── <nombre_config>/
        ├── trial0.pth            # Checkpoint: pesos del modelo, prototipos, class_names, cfg
        ├── trial0_losses.pt      # Historial de pérdidas por época (tensor por cada componente)
        ├── trial1.pth
        ├── trial1_losses.pt
        └── logs/
            ├── trial0.log        # Log detallado del trial: tiempos, métricas por época
            └── trial1.log
```

**Descripción de los archivos:**

| Archivo | Contenido |
|---|---|
| `trial<N>.pth` | Diccionario con `model_state`, `img_proto (C,K,D)`, `txt_proto (C,D)`, `class_names`, `cfg`, `trial`, `seed` |
| `trial<N>_losses.pt` | Diccionario con tensores 1D de longitud `epochs` para cada componente: `loss_id`, `loss_inter`, `loss_intra`, `loss_bias`, `loss_total` |
| `logs/trial<N>.log` | Log del trial: hiperparámetros, tiempos de carga de datos y prototipos, pérdidas cada 10 épocas |

---

### `supreme/evaluate.py`
Evaluación secuencial de SUPREME. Evalúa todos los checkpoints `.pth` de un directorio sobre el conjunto de validación ID y uno o varios conjuntos OOD, calculando FPR95, AUROC y exactitud Top-1.

**Archivos generados:**

```
results/
└── supreme/
    └── <nombre_model_dir>/
        ├── scores.pt             # Scores e índices de clase de todos los trials
        └── evaluation.log        # Log con métricas FPR95/AUROC por conjunto OOD
```

**Descripción de los archivos:**

| Archivo | Contenido |
|---|---|
| `scores.pt` | Diccionario anidado `{trial_name: {"id": {"scores": Tensor(N_id,), "id_labels": Tensor(N_id,)}, "<ood_name>": {"scores": Tensor(N_ood,)}, ...}}` |
| `evaluation.log` | Log de evaluación: métricas FPR95 y AUROC por conjunto OOD, tiempos de cómputo |

---

### `supreme/evaluate_parallel.py`
Versión distribuida de la evaluación. Particiona los conjuntos ID y OOD en `num_chunks` fragmentos y asigna uno a cada nodo. Soporta `DataParallel` multi-GPU. Cada nodo guarda su fragmento de scores de forma independiente.

**Archivos generados:**

```
<save_dir>/
└── scores_node_<chunk_id>.pt     # Scores del fragmento procesado por este nodo
```

**Descripción de los archivos:**

| Archivo | Contenido |
|---|---|
| `scores_node_<chunk_id>.pt` | Diccionario con: `"id_val"` (predicciones y etiquetas), `"id"` (scores e id_labels del fragmento), y una clave por cada dataset OOD con sus scores |

---

### `supreme/baselines.py`
Baselines zero-shot de detección OOD con el backbone CLIP crudo (sin fine-tuning). Calcula simultáneamente MCM y MSP usando prototipos de texto zero-shot obtenidos con la plantilla `"a photo of a {class_name}"`. Opcionalmente puede calcular MMP si se proporcionan checkpoints de SUPREME con `--ckpt`.

**Archivos generados:**

```
<save_dir>/
└── scores_node_<chunk_id>.pt     # Scores del fragmento procesado por este nodo
```

**Descripción de los archivos:**

| Archivo | Contenido |
|---|---|
| `scores_node_<chunk_id>.pt` | Diccionario con: `"id_val"` (predicciones y etiquetas), `"mcm"` y `"msp"` (cada uno con subkeys `"id"` y `"<ood_name>"`), y opcionalmente `"mmp_mean_<K>_shot"` / `"mmp_3d_<K>_shot"` por cada checkpoint provisto |

---

## Configuración

Los experimentos se definen mediante archivos YAML en `config/`. La clase `Config` en `supreme/config.py` carga y valida estos parámetros.

### Tabla de hiperparámetros

| Parámetro | Sección YAML | Descripción | Valores posibles | Por defecto |
|---|---|---|---|---|
| `clip_model` | `clip.model` | Identificador del backbone (HuggingFace Hub o nombre open_clip) | `"hf-hub:imageomics/bioclip-2"`, `"ViT-B-16"`, `"ViT-L-14"`, ... | `"hf-hub:imageomics/bioclip-2"` |
| `clip_pretrained` | `clip.pretrained` | Pesos preentrenados para open_clip (tag o ruta). `null` para modelos HF Hub | `null`, `"openai"`, `"laion2b_s32b_b79k"`, ruta local | `null` |
| `embed_dim` | `clip.embed_dim` | Dimensión de salida del codificador de imagen y texto | `512` (ViT-B/16), `768` (ViT-L/14) | `768` |
| `n_lm` | `clip.n_lm` | Dimensión del embedding de tokens del codificador de texto | `512` (ViT-B/16), `768` (ViT-L/14) | `768` |
| `tau` | `clip.tau` | Temperatura CLIP para el cálculo de logits coseno | `float > 0`, típico: `0.01`–`0.1` | `0.01` |
| `img_size` | `clip.img_size` | Tamaño de la imagen cuadrada de entrada | `224`, `336` | `224` |
| `normalize_mean` | `clip.normalize_mean` | Media de normalización de imagen (por canal RGB) | Lista de 3 floats | `[0.4814, 0.4578, 0.4082]` |
| `normalize_std` | `clip.normalize_std` | Desviación estándar de normalización de imagen (por canal RGB) | Lista de 3 floats | `[0.2686, 0.2613, 0.2758]` |
| `shots` | `train.shots` | Número de imágenes por clase en el conjunto de soporte (K-shot) | Entero positivo, típico: `2`, `4`, `8`, `16` | `16` |
| `epochs` | `train.epochs` | Número de épocas de entrenamiento | Entero positivo | `50` |
| `batch_size` | `train.batch_size` | Tamaño del lote de entrenamiento | Potencia de 2, típico: `16`–`128` | `32` |
| `lr` | `train.lr` | Tasa de aprendizaje para los módulos BPG e ITC | `float`, típico: `1e-4`–`1e-2` | `0.002` |
| `lr_bias` | `train.lr_bias` | Tasa de aprendizaje reducida para μ y σ del sesgo gaussiano | `float`, recomendado ≤ `lr` | `0.0001` |
| `momentum` | `train.momentum` | Momento del optimizador SGD | `0.0`–`0.99` | `0.9` |
| `trials` | `train.trials` | Número de ejecuciones independientes con semillas distintas | Entero positivo | `3` |
| `context_length` | `train.context_length` | Número de tokens de contexto aprendibles del BPG (L) | Entero positivo, típico: `4`–`16` | `16` |
| `alpha` | `train.alpha` | Peso de las pérdidas inter e intra-modal en la pérdida total | `float ≥ 0` | `0.005` |
| `beta` | `train.beta` | Peso de la pérdida de sesgo de dominio en la pérdida total | `float ≥ 0` | `0.1` |
| `seed` | `misc.seed` | Semilla aleatoria base para reproducibilidad (cada trial usa `seed + trial`) | Entero | `42` |
| `num_workers` | `misc.num_workers` | Procesos paralelos para la carga de datos | `0`–`16` | `4` |
| `device` | `misc.device` | Dispositivo de cómputo | `"cuda"`, `"cpu"` | `"cuda"` |
| `text_encode_chunk_size` | `misc.text_encode_chunk_size` | Máximo de secuencias por chunk en el codificador de texto (evita OOM) | Entero positivo, típico: `64`–`512` | `256` |
| `id_train` | `datasets.id_train` | Dataset de entrenamiento (conjunto de soporte few-shot) | `{source, split, samples_per_class, idx_file}` | — |
| `id_val` | `datasets.id_val` | Dataset de validación in-distribution | `{source, split, samples_per_class}` | — |
| `id_test` | `datasets.id_test` | Dataset de test in-distribution | `{source, split, samples_per_class}` | — |
| `ood_test` | `datasets.ood_test` | Dataset(s) out-of-distribution para evaluación. Puede ser un bloque único o lista | `{source, split, samples_per_class}` o lista de ellos | `[]` |
| `score` | `evaluation.score` | Función de puntuación OOD usada en evaluación | `"gmp"`, `"mmp"`, `"mcm_txt"`, `"mcm_img"` | `"gmp"` |
| `mean_img_prototypes` | `evaluation.mean_img_prototypes` | Si `true`, promedía los K shots de cada clase en un único prototipo `(C,D)`. Si `false`, mantiene todos los shots `(C,K,D)` para búsqueda k-NN | `true`, `false` | `false` |

**Campos de `DatasetConfig`:**

| Campo | Descripción | Ejemplo |
|---|---|---|
| `source` | Ruta HuggingFace Hub o directorio local | `"project-oceania/planktonzilla_only_plankton"` |
| `split` | Split del dataset | `"train"`, `"validation"`, `"test"` |
| `samples_per_class` | Límite de muestras por clase. `null` para usar todas | `10`, `null` |
| `idx_file` | Ruta a JSON con índices preseleccionados (generado por `oar_kmeans.sh`). Sobreescribe el muestreo aleatorio | `"data/indices_16shot.json"`, `null` |

---

## Flujo de trabajo con OAR

Los scripts `oar_*.sh` son trabajos para el planificador HPC **OAR**. El flujo completo recomendado es el siguiente:

```
1. oar_baseline.sh  →  2. oar_kmeans.sh  →  3. oar_optuna_optimizer.sh  →  4. oar_train.sh  →  5. oar_parallel_eval.sh
```

---

### 1. `oar_baseline.sh` — Evaluación baseline zero-shot

Lanza `supreme.baselines` en paralelo sobre **4 nodos** (20 chunks, batch 256). Evalúa el backbone CLIP crudo sin fine-tuning, calculando MCM y MSP con prototipos de texto zero-shot. Sirve como referencia para medir cuánto mejora SUPREME respecto al modelo base.

**Salida:** `results/baselines/<config>/scores_node_*.pt`

---

### 2. `oar_kmeans.sh` — Selección few-shot de muestras

Lanza `scripts.fs_selection` en **1 nodo**. Aplica K-Means sobre los embeddings del conjunto de entrenamiento para seleccionar las `shots` muestras más representativas por clase. El resultado es un archivo JSON con los índices seleccionados, que se referencia luego en el YAML como `id_train.idx_file`.

**Salida:** `data/<config>_<shots>shot_indices.json`

---

### 3. `oar_optuna_optimizer.sh` — Optimización de hiperparámetros

Lanza `optimize_hyperparams` como **job array** (2 nodos). En cada nodo detecta automáticamente las GPUs disponibles y lanza un worker de Optuna por GPU en paralelo (`CUDA_VISIBLE_DEVICES=$i &`). Todos los workers comparten la misma base de datos Optuna. Con la configuración por defecto (2 nodos × 2 GPUs × 5 trials), se ejecutan **20 trials** en total para encontrar los mejores hiperparámetros (`lr`, `alpha`, `beta`, etc.).

**Salida:** base de datos Optuna compartida; el mejor trial se usa para crear el YAML definitivo.

---

### 4. `oar_train.sh` — Entrenamiento

Lanza `supreme.train` en **1 nodo** con el YAML del mejor trial encontrado por Optuna. Entrena el modelo completo (BPG + ITC sobre backbone congelado) y guarda los checkpoints con los prototipos cacheados.

**Salida:** `models/supreme/<config>/trial0.pth`, `trial0_losses.pt`, `logs/trial0.log`

---

### 5. `oar_parallel_eval.sh` — Evaluación paralela

Lanza `supreme.evaluate_parallel` distribuido sobre **2 nodos** (20 chunks, batch 256). Cada nodo procesa su fragmento del conjunto ID y los conjuntos OOD, y guarda sus scores parciales. Los archivos resultantes se combinan en un paso posterior para calcular las métricas finales (FPR95, AUROC).

**Salida:** `results/final/<config>/scores_node_*.pt`

---



