"""High-level builders for constructing complex source arrangements.

Provides convenience functions to build magpylib Collections from
engineering-level specifications (solenoid sections, coil arrays, etc.)
without manually creating individual source objects.
"""

from .solenoid import solenoid_collection  # noqa: F401
