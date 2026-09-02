class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def scalar_summary(self, tag, value, step):
        pass