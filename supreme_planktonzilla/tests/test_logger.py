"""Tests for utils/logger.py — ExperimentLogger."""

import logging
import os
import time

import pytest

from utils.logger import ExperimentLogger


def test_logger_creates_with_stream_handler():
    logger = ExperimentLogger(name="test.stream")
    assert any(
        isinstance(h, logging.StreamHandler) for h in logger.logger.handlers
    )


def test_logger_no_duplicate_handlers():
    """Creating two instances with the same name should not add extra handlers."""
    name = "test.no_duplicates"
    logger1 = ExperimentLogger(name=name)
    n_handlers = len(logger1.logger.handlers)
    logger2 = ExperimentLogger(name=name)
    assert len(logger2.logger.handlers) == n_handlers


def test_logger_info_warning_error(capfd):
    logger = ExperimentLogger(name="test.levels")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    out = capfd.readouterr().err  # logging goes to stderr by default
    assert "info message" in out
    assert "warning message" in out
    assert "error message" in out


def test_add_file_handler_creates_file(tmp_path):
    log_path = str(tmp_path / "subdir" / "run.log")
    logger = ExperimentLogger(name="test.file_handler")
    logger.add_file_handler(log_path)
    logger.info("hello file")

    # Flush handlers
    for h in logger.logger.handlers:
        h.flush()

    assert os.path.exists(log_path)
    content = open(log_path).read()
    assert "hello file" in content


def test_add_file_handler_creates_parent_dirs(tmp_path):
    log_path = str(tmp_path / "a" / "b" / "c" / "run.log")
    logger = ExperimentLogger(name="test.nested_dirs")
    logger.add_file_handler(log_path)
    assert os.path.exists(os.path.dirname(log_path))


def test_timer_start_end(capfd):
    logger = ExperimentLogger(name="test.timer")
    logger.start_timer("data_loading")
    time.sleep(0.05)
    logger.end_timer("data_loading")
    out = capfd.readouterr().err
    assert "data loading" in out   # underscores replaced with spaces
    assert "elapsed" in out


def test_timer_end_without_start_warns(capfd):
    logger = ExperimentLogger(name="test.timer_warn")
    logger.end_timer("nonexistent_timer")
    out = capfd.readouterr().err
    assert "WARNING" in out or "never started" in out


def test_timer_removed_after_end():
    logger = ExperimentLogger(name="test.timer_cleanup")
    logger.start_timer("phase")
    logger.end_timer("phase")
    assert "phase" not in logger._timers
