"""
(c) Inria

Python logger factory used throughout the package.
"""

import logging

# from pytorch_lightning.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Return a standard `logging.Logger` for the given name.

    Thin wrapper over `logging.getLogger`; the commented-out block below preserves a former
    `rank_zero_only` pattern intended to avoid duplicated log lines across multi-GPU processes.

    Args:
        name: Logger name, typically the caller's ``__name__``.

    Returns:
        The named `logging.Logger`.
    """
    logger = logging.getLogger(name)
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # logging_levels = (
    #     "debug",
    #     "info",
    #     "warning",
    #     "error",
    #     "exception",
    #     "fatal",
    #     "critical",
    # )
    # for level in logging_levels:
    #    setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
