import logging
import sys

# ANSI escape codes for colors
COLOR_CODES = {
    'DEBUG': '\033[92m',    # Green
    'INFO': '\033[94m',     # Blue
    'WARNING': '\033[93m',  # Yellow
    'ERROR': '\033[91m',    # Red
    'CRITICAL': '\033[95m', # Magenta
    'RESET': '\033[0m'      # Reset to default
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLOR_CODES.get(record.levelname, COLOR_CODES['RESET'])
        message = super().format(record)
        return f"{color}{message}{COLOR_CODES['RESET']}"

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Create a module-level logger instance for use by other modules
logger = get_logger(__name__)