import logging

from .version import VERSION, VERSION_SHORT

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
