from kornia.core import Tensor


class Vec3:
    def __init__(self, data):
        self._data = data

    def __repr__(self) -> str:
        return f"x: {self.x} \ny: {self.y}\nz: {self.z}"

    def __getattr__(self, name: str):
        """Direct access to torch methods."""
        return getattr(self.data, name)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def data(self):
        return self._data

    @property
    def x(self):
        return self.data[..., 0]

    @property
    def y(self):
        return self.data[..., 1]

    @property
    def z(self):
        return self.data[..., 2]

    @classmethod
    def from_raw(cls, x: float, y: float, z: float) -> 'Vec3':
        return cls(Tensor([[x, y, z]]))

    @classmethod
    def zeros(cls):
        return cls(Tensor([[0.0, 0.0, 0.0]]))
