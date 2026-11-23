"""Names of a package loaded on first use (PEP 562).

A package that lists its public names here imports none of its modules when it is
imported, so a light module can import one name of it (``from enginir.preprocessing.recipe
import ...``) without loading scikit-learn and SciPy. The viewer's main window relies on
it: it is shown before the modeling libraries are loaded (``plans/fast-start.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import builtins
from importlib.util import resolve_name
from typing import Any


def lazy_names(package: str, namespace: dict[str, Any],
               names: Mapping[str, str]) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Return ``__getattr__`` and ``__dir__`` for ``package``.

    ``names`` maps each public name to the module it lives in, relative to the package
    (``{"SNV": ".scatter"}``). A name is imported at its first use and then kept in the
    package's ``namespace``.
    """

    def __getattr__(name: str) -> Any:
        module = names.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        # Through ``builtins.__import__`` (not ``import_module``), so that an import hook
        # sees it as it sees an import statement: the viewer's modeling gate does.
        absolute = resolve_name(module, package)
        value = getattr(builtins.__import__(absolute, None, None, ("__name__",), 0), name)
        namespace[name] = value  # resolved once
        return value

    def __dir__() -> list[str]:
        return sorted({*namespace, *names})

    return __getattr__, __dir__
