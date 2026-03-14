import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logger(name: str):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Logging folder setup ---
    logging_folder = Path(__file__).parent.parent / "Logs"
    logging_folder.mkdir(parents=True, exist_ok=True)

    # --- File Handlers ---
    file_handler = logging.FileHandler(logging_folder / f"{name}.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    global_file_handler = logging.FileHandler(
        logging_folder / "log.log", mode="a", encoding="utf-8"
    )
    global_file_handler.setLevel(logging.DEBUG)

    # --- Formatter for file handlers ---
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    global_file_handler.setFormatter(file_formatter)

    # --- Rich Console Handler ---
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,  # This makes the file paths clickable in supported terminals (like Zed/PyCharm/VSCode)
        markup=True,
    )
    console_handler.setLevel(logging.INFO)

    # --- Add all handlers ---
    logger.addHandler(file_handler)
    logger.addHandler(global_file_handler)
    logger.addHandler(console_handler)

    return logger
