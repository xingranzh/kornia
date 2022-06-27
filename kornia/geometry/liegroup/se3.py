from kornia import eye_like
from kornia.core import concatenate

from ._utils import squared_norm
from .so3 import So3
from .vector import Vec3


class Se3:
    def __init__(self, so3: So3, position: Vec3) -> None:
        assert isinstance(so3, So3)
        assert isinstance(position, Vec3)
        self._so3 = so3
        self._position = position

    @property
    def data(self):
        """return the Bx4x4 matrix."""
        return self.matrix()

    def __getattr__(self, name: str):
        """Direct access to torch methods."""
        return getattr(self.data, name)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def position(self):
        return self._position

    @property
    def rotation(self):
        return self._so3

    # return the full Bx4x4 matrix for convenience later multiply points
    def matrix(self):
        from kornia.geometry import convert_affinematrix_to_homography3d

        matrix33 = self._so3.matrix()  # Bx3x3
        matrix34 = concatenate([matrix33, self.position[..., None]], -1)  # Bx3x4
        return convert_affinematrix_to_homography3d(matrix34)  # Bx4x4

    @classmethod
    def exp(cls, v) -> 'Se3':
        """exponential map."""
        upsilon, omega = v[..., :3], v[..., 3:]
        so3, Omega = So3.exp(omega), So3.hat(omega)
        Omega_sq = Omega * Omega
        theta = squared_norm(omega).sqrt()
        V = (
            eye_like(3, theta)
            + (1 - theta.cos()) / theta.pow(2) * Omega
            + (theta - theta.sin()) / theta.pow(3) * Omega_sq
        )
        return Se3(so3, V * upsilon)

    def log(self):
        # import pdb;pdb.set_trace()
        omega = self._so3.log()
        theta = squared_norm(omega).sqrt()
        Omega = So3.hat(omega)

        half_theta = 0.5 * theta

        V_inv = (
            eye_like(3, theta)
            - 0.5 * Omega
            + (1 - theta * half_theta.cos() / (2 * half_theta.sin())) / (theta * theta) * (Omega * Omega)
        )
        upsilon = V_inv * self._position

        return concatenate([upsilon, omega[..., None]], -1)

    @classmethod
    def uninitialised(cls):
        return cls.identity()

    @classmethod
    def identity(cls):
        return cls(So3.identity(), Vec3.zeros())

    def __repr__(self) -> str:
        return f"position:\n{self.position}\nrotation:\n{self.rotation}"
