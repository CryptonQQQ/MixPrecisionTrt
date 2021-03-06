# -*-coding:utf-8-*-
import os
import logging
from logging import handlers


# make sure the exist of filepath
def mkdir(path):
    if not os.path.exists(path):
        os.system(r"touch {}".format(path))  # 调用系统命令行来创建文件
        logging.info("'{}'file path did not exits and has been created successfully.".format(path))
    else:
        logging.info("'{}' exits and loads successfully.".format(path))


class Logger:
    """
    This class is used to write log information.
    Log levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
    """
    log_level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, filename, level='debug', when='D', back_cnt=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d]'
                                                                          ' - %(levelname)s: %(message)s'):
        mkdir(filename)
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.log_level.get(level))
        format_str = logging.Formatter(fmt)
        # print log on screen
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(format_str)
        # write log to file
        file_handler = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=back_cnt,
                                                         encoding='utf-8')
        file_handler.setFormatter(format_str)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)
