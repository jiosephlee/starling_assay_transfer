"""Registry of available task specs, keyed by ``TaskSpec.name``."""

from __future__ import annotations

from .base import TaskSpec
from .oral_bioavailability import OralBioavailabilityTaskSpec

_TASKS: dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> None:
    if not spec.name:
        raise ValueError("task spec must define a non-empty name")
    _TASKS[spec.name] = spec


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known tasks: {sorted(_TASKS)}") from None


def available() -> list[str]:
    return sorted(_TASKS)


# Register built-in specs. Fa/Fg/Fh are added as their modules land.
register(OralBioavailabilityTaskSpec())

try:  # optional until the module exists
    from .fa import FaTaskSpec

    register(FaTaskSpec())
except ImportError:
    pass

try:
    from .fg import FgTaskSpec

    register(FgTaskSpec())
except ImportError:
    pass

try:
    from .fh import FhTaskSpec

    register(FhTaskSpec())
except ImportError:
    pass
