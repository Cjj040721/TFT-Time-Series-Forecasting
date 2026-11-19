"""
Author     : Cao Junjia

Unified logger: writes to console + file simultaneously.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger("train", log_dir="results/logs")
    logger.info("epoch 1 done")
"""
import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "results/logs", level=logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:          # Avoid duplicate handlers on reload
        logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                             datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
