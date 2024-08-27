from env import VERBOSE


def verbose_print(*str):
    if VERBOSE:
        print(*str)
