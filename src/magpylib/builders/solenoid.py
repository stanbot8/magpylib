"""Build magpylib Collections from solenoid section specifications.

A solenoid section is a rectangular cross-section winding with uniform
current density, described by inner/outer radius, axial position, length,
and turn counts. This is the standard building block for MRI magnets,
accelerator magnets, and other multi-coil superconducting systems.

This builder converts engineering-level section specs into magpylib
Circle source collections, with automatic sampling for performance.

Example
-------
>>> from magpylib.builders import solenoid_collection
>>> sections = [
...     {"r_inner": 0.4, "r_outer": 0.45, "z_center": 0.0,
...      "length": 0.3, "n_turns": 100, "n_layers": 50},
... ]
>>> coll = solenoid_collection(sections, current=200.0)
>>> B = coll.getB((0, 0, 0))
"""

import numpy as np
import magpylib as magpy


def solenoid_collection(
    sections: list[dict],
    current: float,
    max_samples_r: int = 6,
    max_samples_z: int = 10,
) -> magpy.Collection:
    """Build a Collection of Circle sources from solenoid section dicts.

    Each section is modeled as a grid of circular current loops
    distributed over its rectangular cross-section. For sections with
    many turns, the loops are sampled and the current is scaled to
    preserve the total ampere-turns.

    Parameters
    ----------
    sections : list of dict
        Each dict describes one solenoid section:
            r_inner : float
                Inner winding radius [m].
            r_outer : float
                Outer winding radius [m].
            z_center : float
                Axial center position [m].
            length : float
                Axial winding length [m].
            n_turns : int
                Number of turns in the axial direction.
            n_layers : int
                Number of turns in the radial direction.
            polarity : float, optional
                Current direction multiplier (+1 or -1). Default +1.
    current : float
        Transport current [A].
    max_samples_r : int
        Maximum radial sample points per section.
    max_samples_z : int
        Maximum axial sample points per section.

    Returns
    -------
    magpylib.Collection
        Collection of Circle sources representing the solenoid system.
    """
    sources = []

    for sec in sections:
        polarity = sec.get("polarity", 1.0)
        I = current * polarity
        total_turns = sec["n_turns"] * sec["n_layers"]

        n_r = min(sec["n_layers"], max_samples_r)
        n_z = min(sec["n_turns"], max_samples_z)
        n_sampled = max(n_r * n_z, 1)
        scale = total_turns / n_sampled

        radii = np.linspace(sec["r_inner"], sec["r_outer"], max(n_r, 1))
        if n_z == 1:
            z_positions = np.array([sec["z_center"]])
        else:
            z_positions = np.linspace(
                sec["z_center"] - sec["length"] / 2,
                sec["z_center"] + sec["length"] / 2,
                n_z,
            )

        for R in radii:
            for z0 in z_positions:
                loop = magpy.current.Circle(
                    current=I * scale,
                    diameter=2 * R,
                )
                loop.position = (0, 0, z0)
                sources.append(loop)

    return magpy.Collection(*sources)
