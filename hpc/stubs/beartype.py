"""
Stub that replaces beartype with a no-op decorator for HPC use.

The TFV6 codebase has some invalid string annotations (e.g. 'torch.Union')
that newer beartype versions reject at import time. Disabling beartype on
the HPC avoids these errors and removes runtime type-checking overhead.
"""


def beartype(obj=None, **kwargs):
    """No-op: return the decorated object unchanged."""
    if obj is None:
        # Called as @beartype(conf=...) — return a no-op decorator
        return lambda fn: fn
    if callable(obj):
        return obj
    return lambda fn: fn


class BeartypeConf:
    def __init__(self, **kwargs):
        pass


class BeartypeStrategy:
    O0 = O1 = O2 = O3 = None


class roar:
    class BeartypeException(Exception): pass
    class BeartypeDecorHintForwardRefException(Exception): pass
    class BeartypeDecorHintException(Exception): pass
