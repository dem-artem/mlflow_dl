import logging

logger = logging.getLogger(__package__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
