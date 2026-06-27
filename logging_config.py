"""Shared logging configuration for the pipeline monitor.

Phase 5: Replaced standard logging with loguru for structured logging.
All agents import `logger` from this module. loguru is a drop-in replacement:
logger.info(), logger.warning(), logger.error(), etc. all work identically.
"""

import sys

from loguru import logger

# Remove the default loguru handler (stderr) so we control output
logger.remove()

# Console handler (stdout) — matches the previous formatting
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    colorize=True,
)

# File handler — append to pipeline.log
logger.add(
    "pipeline.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    level="INFO",
    rotation="10 MB",
    retention="30 days",
    compression="gz",
)
