"""Dimension metadata from docs/model_spec.md sections 2.1 and 4.1.

Coordinate identities, arrays, parameter layouts, and missing-data handling
are deliberately left to later issues and the specification's open questions.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelDimensions:
    """Nonnegative Python integer counts at one observation time t.

    All counts are supplied by the caller, with no defaults or relationships
    imposed across times. Zero counts are representable metadata; they do not
    prescribe filter behavior for missing observations or empty state blocks.
    """

    n_p_t: int
    n_c_t: int
    n_u_t: int
    n_z_t: int

    def __post_init__(self) -> None:
        for field in fields(self):
            count = getattr(self, field.name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{field.name} must be a Python integer")
            if count < 0:
                raise ValueError(f"{field.name} must be nonnegative")

    @property
    def n_s_t(self) -> int:
        """Systematic count: n_s_t = n_p_t + n_c_t."""
        return self.n_p_t + self.n_c_t

    @property
    def n_x_t(self) -> int:
        """Full state count: n_x_t = n_s_t + n_u_t."""
        return self.n_s_t + self.n_u_t
