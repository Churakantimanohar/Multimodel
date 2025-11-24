import logging

_DEF_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "menthheath", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_DEF_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
