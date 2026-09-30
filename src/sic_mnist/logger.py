import sys
import logging

# CRITICAL = 50
# FATAL = CRITICAL
# ERROR = 40
# WARNING = 30
# WARN = WARNING
# INFO = 20
# DEBUG = 10
# NOTSET = 0


# ANSI 颜色码
COLORS = {
    "DEBUG":    "\033[36m",   # 青色
    "INFO":     "\033[32m",   # 绿色
    "WARNING":  "\033[33m",   # 黄色
    "ERROR":    "\033[31m",   # 红色
    "CRITICAL": "\033[1;31m", # 加粗红
}
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, "")
        original = record.levelname
        record.levelname = f"{color}{original}{RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original



def create_logger(name=None, level=logging.INFO):
    name = name or __name__        # 不传就用当前模块名
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler()
    # !!!
    handler.setLevel(logging.NOTSET)

    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    if sys.stdout.isatty():           # 终端才上色
        handler.setFormatter(ColorFormatter(fmt, datefmt="%H:%M:%S"))
    else:
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger.addHandler(handler)
    return logger

_internal_logger = create_logger(level=logging.INFO)


class LogProxy:

    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    """代理类，让 log.info(a,b,c) 支持多参数"""
    def debug(self, *args):
        _internal_logger.debug(" ".join(map(str, args)))

    def info(self, *args):
        _internal_logger.info(" ".join(map(str, args)))

    def warning(self, *args):
        _internal_logger.warning(" ".join(map(str, args)))

    def error(self, *args):
        _internal_logger.error(" ".join(map(str, args)))

    def critical(self, *args):
        _internal_logger.critical(" ".join(map(str, args)))

    def set_level(self, level):
        """外部修改日志级别: set_level(logging.DEBUG)"""
        _internal_logger.setLevel(level)
        for h in _internal_logger.handlers:
            h.setLevel(level)

    def get_raw(self):
        return _internal_logger


# 对外别名，方便其他文件 import
log = LogProxy()

__all__ = ["log"]