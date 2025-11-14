import logging
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """Adds colors and formatting for console logs."""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"
    
class CustomLogger(logging.Logger):
    """Custom logger that automatically appends a traceback on errors."""

    def error(self, msg, *args, exc_info=True, tb_lines_count=25, **kwargs):
        """Append the last N lines of a traceback if available."""
        if exc_info:
            tb_lines = traceback.format_exc().splitlines()
            if tb_lines and tb_lines[0].startswith("Traceback"):
                header = tb_lines[:1]
                tail = tb_lines[-tb_lines_count:] if len(tb_lines) > tb_lines_count else tb_lines
                short_tb = "\n".join(header + tail)
                msg = f"{msg}\n{short_tb}"
        super().error(msg, *args, **kwargs)

    def exception(self, msg, *args, tb_lines_count=25, **kwargs):
        """Ensure traceback always appears for .exception()."""
        tb_lines = traceback.format_exc().splitlines()
        if tb_lines and tb_lines[0].startswith("Traceback"):
            header = tb_lines[:1]
            tail = tb_lines[-tb_lines_count:] if len(tb_lines) > tb_lines_count else tb_lines
            short_tb = "\n".join(header + tail)
            msg = f"{msg}\n{short_tb}"
        super().exception(msg, *args, **kwargs)


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Creates and returns a configured, colorful logger instance.

    Args:
        name (str): Logger name, usually __name__ of the caller module.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    # Create logs directory under project root
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "atlasmind.log"

    # Base format
    base_format = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # File formatter (no color)
    file_formatter = logging.Formatter(fmt=base_format, datefmt=date_format)

    # Console formatter (with color)
    console_formatter = ColorFormatter(fmt=base_format, datefmt=date_format)

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=2 * 1024 * 1024,  # 2 MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Set log level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    return logger
