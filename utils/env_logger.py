import logging
from config.settings import LOGGER, LOG_LEVEL, LOG_FILE

def setup_logger() -> logging.Logger:
    log = logging.getLogger(LOGGER)
    log.setLevel(LOG_LEVEL)

    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(LOG_LEVEL)

    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(ch)

    return log

env_logger = setup_logger()
