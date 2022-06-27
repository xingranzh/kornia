from math import pi
from typing import Optional

import torch
from torch import cos, rand, sin, sqrt

from kornia.core import Tensor, concatenate, stack


class Quaternion:
    def __init__(self, real: Tensor, vec: Tensor) -> None:
        self._real = real
        self._vec = vec

    def __repr__(self) -> str:
        return f"real: {self.real} \nvec: {self.vec}"

    def __getattr__(self, name: str):
        """Direct access to torch methods."""
        return getattr(self.data, name)

    def __getitem__(self, idx):
        return self.data[idx]

    def __neg__(self) -> 'Quaternion':
        return Quaternion(-self.real, -self.vec)

    def __add__(self, right: 'Quaternion') -> 'Quaternion':
        assert isinstance(right, Quaternion)
        return Quaternion(self.real + right.real, self.vec + right.vec)

    @property
    def real(self) -> Tensor:
        return self._real

    @property
    def vec(self) -> Tensor:
        return self._vec

    @property
    def data(self) -> Tensor:
        return concatenate([self.real, self.vec], -1)

    def matrix(self):
        from kornia.geometry.conversions import QuaternionCoeffOrder, quaternion_to_rotation_matrix

        return quaternion_to_rotation_matrix(self.data, order=QuaternionCoeffOrder.WXYZ)

    @classmethod
    def identity(cls):
        real_t = Tensor([[1.0]])
        imag_t = Tensor([[0.0, 0.0, 0.0]])
        return cls(real_t, imag_t)

    @classmethod
    def from_raw(cls, real: float, imag_x: float, imag_y: float, imag_z: float) -> 'Quaternion':
        real_t = Tensor([[real]])
        imag_t = Tensor([[imag_x, imag_y, imag_z]])
        return cls(real_t, imag_t)

    @classmethod
    def random(
        cls, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> 'Quaternion':
        """Generate a random unit quaternion.

        Uniformly distributed across the rotation space
        As per: http://planning.cs.uiuc.edu/node198.html
        """
        r1, r2, r3 = rand(3, batch_size, device=device, dtype=dtype)
        q1 = sqrt(1.0 - r1) * (sin(2 * pi * r2))
        q2 = sqrt(1.0 - r1) * (cos(2 * pi * r2))
        q3 = sqrt(r1) * (sin(2 * pi * r3))
        q4 = sqrt(r1) * (cos(2 * pi * r3))
        return cls(q1, stack((q2, q3, q4), -1))

    def norm(self) -> Tensor:
        q = concatenate((self.real[..., None], self.vec), -1)
        return q.norm(p=2, dim=-1)

    def conj(self) -> 'Quaternion':
        return Quaternion(self.real, -self.vec)
