import logging
import os
import time


# ANSI escape codes for colors in terminal
class TerminalColor:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: TerminalColor.CYAN,
        logging.INFO: TerminalColor.GREEN,
        logging.WARNING: TerminalColor.YELLOW,
        logging.ERROR: TerminalColor.RED,
        logging.CRITICAL: TerminalColor.RED + TerminalColor.WHITE
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelno)
        reset_color = TerminalColor.RESET
        message = super(ColorFormatter, self).format(record)
        return level_color + message + reset_color


log_path = os.path.join(os.path.dirname(__file__), 'logs')
# if not os.path.exists(log_path):
#     os.mkdir(log_path)


class Log(object):
    def __init__(self, log_name):
        self.log_name = log_name
        self.logname = os.path.join(log_path, '%s_%s.log' % (self.log_name, time.strftime('%Y_%m_%d')))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # formatter = ColorFormatter('[%(asctime)s] - %(filename)s] - %(levelname)s: %(message)s')
        formatter = ColorFormatter('[%(levelname)s] %(message)s')

        # # Create file handler
        # fh = logging.FileHandler(self.logname, 'a', encoding='utf-8')
        # fh.setLevel(logging.DEBUG)
        # fh.setFormatter(formatter)

        # Create console handler with color support
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.ch.setFormatter(formatter)

        # self.logger.addHandler(fh)
        self.logger.addHandler(self.ch)
    def changeLevel(self,level):
        # self.logger.setLevel(level)
        self.logger.setLevel(level)

    def debug(self, message):
        self.logger.debug(message)
        

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == "__main__":
    log = Log('Confluence')
    log.info("---测试开始----")
    log.info("操作步骤1,2,3")
    log.warning("----测试结束----")
    log.error("----测试中有错误----")
    log.critical("----测试中有致命错误----")

