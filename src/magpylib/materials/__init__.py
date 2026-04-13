"""Material property models for superconducting and magnetic sources.

Provides temperature- and field-dependent material properties that
extend magpylib's static source models with realistic physics.
"""

from .hts_tape import (  # noqa: F401
    HTSTape,
    SUPERPOWER_SCS4050,
    SUPERPOWER_SCS12050,
    FUJIKURA_FESC,
)
