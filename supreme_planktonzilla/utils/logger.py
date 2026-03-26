"""
Logger de experimentos con salida con timestamp y funcionalidad de temporizador.

Combina el módulo logging de Python con medición de tiempos para registrar
el progreso de los experimentos tanto en consola como en archivo .log.
"""

import logging
import os
import time


class ExperimentLogger:
    """
    Logger para experimentos de detección OOD.

    Combina el módulo logging de Python con funcionalidad de temporizador para
    proporcionar mensajes con timestamp y medición de tiempo de cada etapa
    del experimento. Puede escribir simultáneamente a consola y a archivo .log.

    Args:
        name (str): Nombre del logger. Por defecto 'ood_experiment'.
        level (int): Nivel de logging. Por defecto logging.INFO.

    Example:
        >>> logger = ExperimentLogger()
        >>> logger.info("Starting experiment")
        [2026-02-12 23:30:00] [INFO] Starting experiment
        >>> logger.start_timer("data_loading")
        [2026-02-12 23:30:00] [INFO] Timer 'data loading' started
        >>> logger.end_timer("data_loading")
        [2026-02-12 23:30:05] [INFO] Timer 'data loading' elapsed: 0.08 minutes
    """

    def __init__(self, name: str = "ood_experiment", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Evitar handlers duplicados si el logger ya existe
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self._timers = {}

    def add_file_handler(self, log_path: str) -> None:
        """
        Añade un handler de archivo para guardar los logs en disco.

        Crea los directorios necesarios si no existen. A partir de esta
        llamada, todos los mensajes se escriben tanto en consola como en
        el archivo indicado.

        Args:
            log_path (str): Ruta completa al archivo .log de destino.
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setLevel(self.logger.level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def info(self, msg: str) -> None:
        """Registra un mensaje de nivel INFO."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Registra un mensaje de nivel WARNING."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Registra un mensaje de nivel ERROR."""
        self.logger.error(msg)

    def start_timer(self, name: str) -> None:
        """
        Inicia un temporizador con nombre y registra el evento.

        Args:
            name (str): Identificador del temporizador. Los guiones bajos
                        se muestran como espacios en el mensaje.
        """
        self._timers[name] = time.time()
        display_name = " ".join(name.split("_"))
        self.logger.info(f"Timer '{display_name}' started")

    def end_timer(self, name: str) -> None:
        """
        Finaliza un temporizador, registra el tiempo transcurrido y lo elimina.

        Args:
            name (str): Identificador del temporizador. Debe coincidir con
                        una llamada previa a start_timer().
        """
        if name in self._timers:
            elapsed = time.time() - self._timers[name]
            display_name = " ".join(name.split("_"))
            self.logger.info(
                f"Timer '{display_name}' elapsed: {elapsed / 60:.2f} minutes"
            )
            self._timers.pop(name)
        else:
            self.logger.warning(f"Timer '{name}' was never started")
