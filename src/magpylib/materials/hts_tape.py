"""HTS (High-Temperature Superconductor) coated conductor tape models.

Provides critical current Ic(T, B, theta) parameterization for REBCO
(Rare Earth Barium Copper Oxide) tapes from major manufacturers.

The critical current model uses the standard parameterization:

    Ic(T, B, theta) = Ic0 * f(T) * g(B, theta)

where:
    f(T) = (1 - T/Tc)^alpha
    g(B, theta) = (1 + sqrt((B*cos(theta)/Bc)^2 + (B*sin(theta)/Bab)^2))^(-beta)
    theta = angle between applied field and tape c-axis (0 = perpendicular)

Parameters are fitted to manufacturer data sheets. The model is
self-calibrating: specify ``Ic_at_77K_sf`` (the manufacturer's
self-field Ic at 77 K) and ``Ic0`` is computed automatically.

Example
-------
>>> from magpylib.materials import HTSTape, SUPERPOWER_SCS12050
>>> tape = SUPERPOWER_SCS12050
>>> tape.Ic(65.0, 2.0, theta=0)  # 65 K, 2 T perpendicular
361.7...

References
----------
.. [1] Senatore et al., Supercond. Sci. Technol. 27 (2014) 103001
.. [2] Hilton et al., Supercond. Sci. Technol. 19 (2006) 1145
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class HTSTape:
    """REBCO coated conductor tape properties and Ic model.

    Parameters
    ----------
    width : float
        Tape width [m].
    thickness : float
        Total tape thickness [m] (substrate + buffer + REBCO + stabilizer).
    rebco_thickness : float
        REBCO superconducting layer thickness [m].
    Ic_at_77K_sf : float
        Critical current at 77 K, self-field [A] (manufacturer spec).
    Tc : float
        Critical temperature [K].
    alpha : float
        Temperature scaling exponent.
    Bc : float
        c-axis characteristic field [T].
    Bab : float
        ab-plane characteristic field [T].
    beta : float
        Field scaling exponent.
    n_value : float
        Power-law index for the V-I characteristic.
    """

    # Geometry
    width: float = 4.0e-3
    thickness: float = 0.1e-3
    rebco_thickness: float = 1.5e-6

    # Ic model
    Ic0: float = 0.0
    Ic_at_77K_sf: float = 120.0
    Tc: float = 92.0
    alpha: float = 1.5
    Bc: float = 0.8
    Bab: float = 5.0
    beta: float = 0.7
    n_value: float = 30.0

    # Mechanical
    tensile_strength: float = 700e6
    yield_strength: float = 550e6
    elastic_modulus: float = 150e9
    density: float = 8200.0

    # Thermal
    cp_77K: float = 200.0
    k_along: float = 400.0
    k_perp: float = 1.0
    rho_normal: float = 2e-6

    def __post_init__(self):
        if self.Ic0 == 0.0 and self.Ic_at_77K_sf > 0:
            f_77 = (1.0 - 77.0 / self.Tc) ** self.alpha
            self.Ic0 = self.Ic_at_77K_sf / f_77

    def Ic(self, T: float, B: float, theta: float = np.pi / 2) -> float:
        """Critical current at given temperature, field, and angle.

        Parameters
        ----------
        T : float
            Temperature [K].
        B : float
            Applied magnetic field magnitude [T].
        theta : float
            Angle between field and tape c-axis (normal to tape surface).
            theta=0: perpendicular (worst case). theta=pi/2: parallel (best).

        Returns
        -------
        float
            Critical current [A]. Returns 0 if T >= Tc.
        """
        if T >= self.Tc:
            return 0.0
        f_T = (1.0 - T / self.Tc) ** self.alpha
        B_eff = np.sqrt(
            (B * np.cos(theta) / self.Bc) ** 2
            + (B * np.sin(theta) / self.Bab) ** 2
        )
        g_B = (1.0 + B_eff) ** (-self.beta)
        return self.Ic0 * f_T * g_B

    def Ic_array(
        self,
        T: np.ndarray,
        B: np.ndarray,
        theta: np.ndarray | float = np.pi / 2,
    ) -> np.ndarray:
        """Vectorized critical current for arrays of T, B, theta."""
        T = np.asarray(T, dtype=float)
        B = np.asarray(B, dtype=float)
        theta = np.broadcast_to(theta, T.shape)
        f_T = np.where(T < self.Tc, (1.0 - T / self.Tc) ** self.alpha, 0.0)
        B_eff = np.sqrt(
            (B * np.cos(theta) / self.Bc) ** 2
            + (B * np.sin(theta) / self.Bab) ** 2
        )
        g_B = (1.0 + B_eff) ** (-self.beta)
        return self.Ic0 * f_T * g_B

    def Jc(self, T: float, B: float, theta: float = np.pi / 2) -> float:
        """Critical current density in the REBCO layer [A/m2]."""
        return self.Ic(T, B, theta) / (self.width * self.rebco_thickness)


# ---------------------------------------------------------------------------
# Predefined tapes from manufacturer data sheets
# ---------------------------------------------------------------------------

SUPERPOWER_SCS4050 = HTSTape(
    Ic_at_77K_sf=120.0,
)
"""SuperPower SCS4050-AP, 4 mm wide."""

SUPERPOWER_SCS12050 = HTSTape(
    width=12.0e-3,
    Ic_at_77K_sf=360.0,
)
"""SuperPower SCS12050, 12 mm wide."""

FUJIKURA_FESC = HTSTape(
    width=4.0e-3,
    Ic_at_77K_sf=150.0,
    Bc=0.6,
    beta=0.65,
)
"""Fujikura FESC, 4 mm wide."""
