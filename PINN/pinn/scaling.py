"""Nondimensionalisation and nondimensional geometry.

    x* = x / D,   u* = u / U,   p* = p / (rho U^2),   Re = rho U D / mu
with D the healthy diameter and U the peak inlet velocity. The steady NS
equations become  (u*.grad)u* + grad p* - (1/Re) lap u* = 0,  div u* = 0.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Scales:
    D: float      # healthy diameter [m]
    U: float      # peak inlet velocity [m/s]
    rho: float
    mu: float

    @property
    def Re(self):
        return self.rho * self.U * self.D / self.mu

    @property
    def P(self):
        return self.rho * self.U ** 2

    @classmethod
    def from_meta(cls, meta):
        prm, geo = meta["params"], meta["geometry"]
        return cls(D=2.0 * geo["radius"], U=prm["u_max"], rho=prm["rho"], mu=prm["mu"])


@dataclass(frozen=True)
class Geometry:
    """Nondimensional channel: x* in [0, length], |y*| <= r(x*, s), healthy R = 0.5."""
    length: float
    R: float
    x_center: float
    half_length: float

    @classmethod
    def from_meta(cls, meta, scales):
        g = meta["geometry"]
        return cls(length=g["length"] / scales.D, R=g["radius"] / scales.D,
                   x_center=g["x_center"] / scales.D, half_length=g["half_length"] / scales.D)

    def key(self):
        return asdict(self)
