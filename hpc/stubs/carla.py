"""
Minimal CARLA stub for offline / HPC environments where the CARLA simulator
is not installed.  Satisfies all `import carla` calls in the TFV6 codebase
without requiring the actual CARLA Python package.

Any attribute access on the module returns a no-op mock class, so type
annotations and module-level constants resolve at import time.  Actual
CARLA simulation calls (e.g. spawning actors) will silently return mocks
and must not be used in offline code paths.
"""


class _Mock:
    """Generic stub: accepts any constructor args, returns mocks for attribute access."""
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return _Mock()
    def __call__(self, *args, **kwargs):
        return _Mock()
    def __iter__(self):
        return iter([])
    def __bool__(self):
        return False


def __getattr__(name):
    """Return a mock class for any CARLA type (Transform, Location, Actor, ...)."""
    return _Mock
