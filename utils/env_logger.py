import logging
from config.settings import LOGGER, LOG_LEVEL, LOG_FILE

def setup_logger() -> logging.Logger:
    log = logging.getLogger(LOGGER)
    log.setLevel('DEBUG')

    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel('DEBUG')

    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(ch)
    log.propagate = False

    return log

env_logger = setup_logger()
