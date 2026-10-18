"""Application logging configuration."""

import logging
import logging.config


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent console logging."""

    normalized_level = level.upper()

    if normalized_level not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        normalized_level = "INFO"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": (
                        "%(asctime)s | %(levelname)s | "
                        "%(name)s | %(message)s"
                    ),
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
            },
            "root": {
                "level": normalized_level,
                "handlers": ["console"],
            },
        }
    )