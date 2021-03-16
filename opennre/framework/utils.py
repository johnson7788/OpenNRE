class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


import logging,os
names = set()
def __setup_custom_logger(name: str, logfile: str) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    #日志格式
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',)

    names.add(name)

    #日志输出到console
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    #日志级别
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    # 日志文件设置
    LOGFILE = os.path.expanduser(logfile)
    file_handler = logging.FileHandler(LOGFILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def get_logger(name: str, logfile=None) -> logging.Logger:
    if logfile is None:
        logfile = f"{name}.log"
    if name in names:
        return logging.getLogger(name)
    else:
        return __setup_custom_logger(name,logfile)