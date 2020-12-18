import logging
from mylib1 import mylib1_do1

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

logger.info('main is runing')
logging.info('runing')
mylib1_do1('Up')
