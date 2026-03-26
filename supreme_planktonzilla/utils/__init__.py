"""
Utilidades comunes para los métodos de detección OOD basados en CLIP.

Este paquete contiene módulos compartidos entre todos los métodos:
  - logger    : ExperimentLogger con soporte para archivo .log y temporizadores
  - datasets  : load_dataset_from_config, CLIPDataset, few_shot_subset
  - metricas  : métricas de evaluación OOD (AUROC, FPR95)
  - io        : carga de configuraciones YAML y guardado de resultados JSON
  - transforms: transformaciones de imagen estándar para CLIP
"""
