"""
Punto de entrada principal del framework de detección OOD basado en CLIP.

Cada método es un submódulo de este proyecto (ej: supreme/) que expone
las funciones run_train(config_path) y run_evaluate(model_dir, config_path).

Uso:
    # Solo entrenar
    python main.py supreme --config config/supreme_default.yaml --train

    # Solo evaluar
    python main.py supreme --config config/supreme_default.yaml \
                           --eval models/supreme/supreme_default

    # Entrenar y luego evaluar en la misma llamada
    python main.py supreme --config config/supreme_default.yaml --train \
                           --eval models/supreme/supreme_default
"""

import argparse
import importlib
import sys


def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos del framework.

    Argumentos
    ----------
    method   : Nombre del método CLIP a ejecutar (ej: 'supreme'). Debe
               corresponder a un submódulo del proyecto que exporte
               run_train() y run_evaluate().
    --config : Ruta al archivo YAML con la configuración del experimento.
               Requerido siempre.
    --train  : Flag (sin valor). Si está presente, ejecuta el entrenamiento.
    --eval   : Ruta a la carpeta donde se guardaron los modelos entrenados.
               Si está presente, ejecuta la evaluación sobre esos modelos.

    Retorna
    -------
    argparse.Namespace : Objeto con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Framework de detección OOD basado en CLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "method",
        type=str,
        help="Nombre del método CLIP a ejecutar (ej: supreme).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="YAML",
        help="Ruta al archivo YAML de configuración del experimento.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Si se incluye, ejecuta el entrenamiento del método.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        metavar="MODEL_DIR",
        help="Carpeta donde se guardaron los modelos entrenados. "
             "Si se incluye, ejecuta la evaluación sobre esos modelos.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Función principal del framework.

    Carga dinámicamente el módulo del método indicado y ejecuta el
    entrenamiento y/o la evaluación según los flags recibidos.
    """
    args = parse_args()

    if not args.train and args.eval is None:
        print("Error: se debe especificar al menos --train o --eval.")
        sys.exit(1)

    # Importar el módulo del método dinámicamente (ej: import supreme)
    try:
        method = importlib.import_module(args.method)
    except ModuleNotFoundError:
        print(f"Error: no se encontró el método '{args.method}'. "
              f"Asegúrate de que existe una carpeta '{args.method}/' con __init__.py.")
        sys.exit(1)

    # Entrenamiento
    if args.train:
        print(f"\n{'='*60}")
        print(f"  Entrenamiento: {args.method}  |  config: {args.config}")
        print(f"{'='*60}\n")
        method.run_train(args.config)

    # Evaluación
    if args.eval is not None:
        print(f"\n{'='*60}")
        print(f"  Evaluación: {args.method}  |  modelos: {args.eval}")
        print(f"{'='*60}\n")
        method.run_evaluate(args.eval, args.config)


if __name__ == "__main__":
    main()
