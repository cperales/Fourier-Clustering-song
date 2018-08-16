import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
