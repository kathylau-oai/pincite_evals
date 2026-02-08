"""Backwards-compatible exports for the eval runner package."""

from . import runner as _runner

# Keep package-level imports stable (including underscored test helpers)
# while moving implementation out of __init__.py.
for export_name, export_value in _runner.__dict__.items():
    if export_name.startswith("__"):
        continue
    globals()[export_name] = export_value

__all__ = [name for name in dir(_runner) if not name.startswith("__")]
