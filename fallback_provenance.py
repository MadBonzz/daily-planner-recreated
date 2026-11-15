"""Record the exact packaged results used after live model failures in this operation."""

from contextlib import contextmanager
from contextvars import ContextVar


_fallbacks = ContextVar("planner_packaged_fallbacks", default=None)


@contextmanager
def capture_fallbacks():
    roles = set()
    parent = _fallbacks.get()
    token = _fallbacks.set(roles)
    try:
        yield roles
    finally:
        _fallbacks.reset(token)
        if parent is not None:
            parent.update(roles)


def record_packaged_fallback(role):
    roles = _fallbacks.get()
    if roles is not None:
        roles.add(role)
