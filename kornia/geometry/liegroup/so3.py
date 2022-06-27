from kornia.core import Tensor

from ._utils import squared_norm
from .quaternion import Quaternion


class So3:
    def __init__(self, q: Quaternion) -> None:
        assert isinstance(q, Quaternion)
        assert len(q.shape) == 2 and q.shape[-1] == 4
        self._q = q

    def __repr__(self) -> str:
        return f"{self.q}"

    def __getattr__(self, name: str):
        """Direct access to torch methods."""
        return getattr(self.data, name)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def q(self):
        return self._q

    @classmethod
    def exp(cls, v) -> 'So3':
        theta_sq = squared_norm(v)
        theta = theta_sq.sqrt()
        return cls(Quaternion((0.5 * theta).cos(), (0.5 * theta).sin().div(theta).mul(v[:])))

    def log(self):
        n = squared_norm(self.q.vec).sqrt()
        return 2 * (n / self.q.real).atan() / n * self.q.vec

    @staticmethod
    def hat(o):
        import torch  # possibly solve with stack

        zeros = torch.zeros_like(o)[..., 0]
        row0 = torch.cat([zeros, -o[..., 2], o[..., 1]], -1)
        row1 = torch.cat([o[..., 2], zeros, -o[..., 0]], -1)
        row2 = torch.cat([-o[..., 1], o[..., 0], zeros], -1)
        return torch.stack([row0, row1, row2], -2)

    def matrix(self):
        """returns matrix representation."""
        # NOTE: this differs from kornia.geometry.quaternion_to_rotation_matrix
        # NOTE: add tests because in kornia is double checked
        return Tensor(
            [
                [
                    [
                        1 - 2 * self.q.vec[..., 1] ** 2 - 2 * self.q.vec[..., 2] ** 2,
                        2 * self.q.vec[..., 0] * self.q.vec[..., 1] - 2 * self.q.vec[..., 2] * self.q[..., 3],
                        2 * self.q.vec[..., 0] * self.q.vec[..., 2] + 2 * self.q.vec[..., 1] * self.q[..., 3],
                    ],
                    [
                        2 * self.q.vec[..., 0] * self.q.vec[..., 1] + 2 * self.q.vec[..., 2] * self.q[..., 3],
                        1 - 2 * self.q.vec[..., 0] ** 2 - 2 * self.q.vec[..., 2] ** 2,
                        2 * self.q.vec[..., 1] * self.q.vec[..., 2] - 2 * self.q.vec[..., 0] * self.q[..., 3],
                    ],
                    [
                        2 * self.q.vec[..., 0] * self.q.vec[..., 2] - 2 * self.q.vec[..., 1] * self.q[..., 3],
                        2 * self.q.vec[..., 1] * self.q.vec[..., 2] + 2 * self.q.vec[..., 0] * self.q[..., 3],
                        1 - 2 * self.q.vec[..., 0] ** 2 - 2 * self.q.vec[..., 1] ** 2,
                    ],
                ]
            ]
        )

    @classmethod
    def identity(cls):
        return cls(Quaternion.identity())
