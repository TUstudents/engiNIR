"""Registry of the viewer's plots and of the plot pages of each workspace.

Every plot of the viewer is registered here, so that one test can check that
each of them renders an empty state and exports PNG, SVG and CSV (Plot Panel
Policy, ``viewer-workspace-ui-contract_v3.md``).
"""

from __future__ import annotations

from ._plot_grid import PlotPage
from ._plot_panel import PlotSpec
from .._checks import log_and_raise

#: All registered plots by id.
PLOT_SPECS: dict[str, PlotSpec] = {}
#: The plot pages of each workspace, in display order.
WORKSPACE_PAGES: dict[str, tuple[PlotPage, ...]] = {}


def register_plot(spec: PlotSpec) -> PlotSpec:
    """Add ``spec`` to the registry; plot ids are unique."""
    if spec.plot_id in PLOT_SPECS:
        log_and_raise("Plot '{}' is already registered", spec.plot_id)
    PLOT_SPECS[spec.plot_id] = spec
    return spec


def register_pages(workspace: str, pages: tuple[PlotPage, ...]) -> None:
    """Set the plot pages of ``workspace``; every slot must name a registered plot."""
    if workspace in WORKSPACE_PAGES:
        log_and_raise("Workspace '{}' already has plot pages", workspace)
    unknown = [plot_id for page in pages for plot_id in page.plot_ids
               if plot_id not in PLOT_SPECS]
    if unknown:
        log_and_raise("Workspace '{}' pages name unregistered plots: {}", workspace, unknown)
    WORKSPACE_PAGES[workspace] = pages


__all__ = ["PLOT_SPECS", "WORKSPACE_PAGES", "register_pages", "register_plot"]
