import logging
import os
import argparse

class ColorFormatter(logging.Formatter):
    """
    Custom formatter that adds color to the log messages.
    """
    # ANSI escape codes for colors and bold
    COLOR_CODES = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m',  # Magenta
    }
    RESET_CODE = '\033[0m'  # Reset color
    BOLD_CODE = '\033[1m'  # Bold text
    CYAN_CODE = '\033[96m'  # Cyan for module name

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record with color.
        
        Args:
            record (logging.LogRecord): Log record to format.
            
        Returns:
            str: Formatted log message.
        """
        # Add color to the log level name
        color = self.COLOR_CODES.get(record.levelname, self.RESET_CODE)
        record.levelname = f"{color}{record.levelname}{self.RESET_CODE}"

        # Make the module name bold and cyan
        record.module = f"{self.CYAN_CODE}{self.BOLD_CODE}{record.module}{self.RESET_CODE}"

        return super().format(record)

def setup_logging(log_level: str = "DEBUG", log_output: str = "../.debugging"):
    """
    Sets up logging with the specified log level.
    
    Args:
        log_level (str): Log level as a string (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    # Convert log level string to numeric level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Set up logging with the custom formatter
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColorFormatter(
            fmt=
            "%(asctime)s [%(levelname)-8s]\t %(module)s: %(message)s (%(filename)s:%(lineno)s)",
            datefmt="%Y-%m-%d %H:%M:%S"))

    if not os.path.exists(log_output):
        os.makedirs(log_output)

    file_handler = logging.FileHandler(os.path.join(log_output,
                                                    "test_log.txt"),
                                       mode='w')
    file_handler.setLevel(log_level.upper())

    logging.basicConfig(level=log_level, handlers=[handler, file_handler])

    # Set third-party loggings to WARNING level or higher. They are annoying.
    for lib in ['matplotlib', 'font_manager', 'PIL']:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Optional: log a confirmation message
    logging.debug(f"Logging initialized with level: {log_level}")
    
    logging.info("Starting pipeline")
    logging.debug("Debugging information")
    logging.warning("Warning message")
    logging.error("Error message")
    logging.critical("Critical error")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Kalman Filter Functionality Test')
    parser.add_argument(
        '--config_file',
        type=str,
        default="./configs/config_uav.yaml",
        help='Path to the configuration file.')
    parser.add_argument(
        '--log_output',
        type=str,
        default='../_debugging/',
        help='Path to the configuration file.')
    parser.add_argument(
        '--use_default_config',
        action='store_true',
        help='Use default configuration')
    parser.add_argument(
        '--log_level',
        default='DEBUG',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level")
    parser.add_argument(
        '--early_stop',
        action='store_true',
        help='Stop the pipeline after the first iteration')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_level="DEBUG", log_output=args.log_output)