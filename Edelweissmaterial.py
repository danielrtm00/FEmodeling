import numpy as np
import numpy.linalg as lin

class Edelweissmaterial():
    """Defines a linear elastic material."""
    
    def __init__(self, materialName: str, materialProperties: np.ndarray):
        """Define the material properties.
        
        Parameters
        ----------
        materialName
            A string identifying the requested material.
        materialProperties
            The numpy array containing the material properties."""

        if materialName == "linearelastic":
            self.E = materialProperties[0]  # set E
            self.v = materialProperties[1]  # set v
        else:
            raise Exception("This material type doesn't exist (yet).")

    def computeStiffness(self, K: np.array, elementType: str, x: np.array, th: float):
        """Compute the stiffness matrix K from a given material for a given element."""

        self.elementtype = elementType
        if self.elementtype[0] in ("Q", "T") and (
            self.elementtype[6:8] in ("PE", "") or (len(self.elementtype) == 7 and self.elementtype[5:7] == "PE")
        ):
            # Ei = elasticity matrix for plane strain for element i (mostly the case for rectangular elements)
            Ei = (
                (self.E * (1 - self.v))
                / ((1 + self.v) * (1 - 2 * self.v))
                * np.array(
                    [
                        [1, self.v / (1 - self.v), 0],
                        [self.v / (1 - self.v), 1, 0],
                        [0, 0, (1 - 2 * self.v) / (2 * (1 - self.v))],
                    ]
                )
            )
        elif self.elementtype[0] in ("Q", "T") and (
            self.elementtype[6:8] == "PS" or (len(self.elementtype) == 7 and self.elementtype[5:7] == "PS")
        ):
            # Ei = elasticity matrix for plane stress for element i
            Ei = self.E / (1 - self.v**2) * np.array([[1, self.v, 0], [self.v, 1, 0], [0, 0, (1 - self.v) / 2]])
        if self.elementtype[0] == "Q":  # it's a 2D quadrilateral element
            # [a] matrix that connects strain and displacement derivatives
            a = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
            # integration points
            if self.elementtype[4:6] in ("4N", "8R", "4") or (
                len(self.elementtype) == 7 and self.elementtype[4] == "4"
            ):  # assume it's normal integration if it's not given
                Nint = 4
                # get all points
                t = 1 / np.sqrt(3) * np.array([-1, -1, 1, 1])  # get t
                s = 1 / np.sqrt(3) * np.array([-1, 1, 1, -1])  # get s
                # get weights
                w = np.array([1, 1, 1, 1])  # weights to all corresponding points
            elif self.elementtype[4:6] == "4R":
                Nint = 1
                t = np.array([0])  # get t
                s = np.array([0])  # get s
                # get weights
                w = np.array([4])  # weights to all corresponding points
            elif self.elementtype[4:6] in ("8N", "4E", "8") or (
                len(self.elementtype) == 7 and self.elementtype[4] == "8"
            ):  # use 9 integration points (for 8 noded element or extended int)
                Nint = 9
                # get all points
                t = np.sqrt(0.6) * np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])  # get t
                s = np.sqrt(0.6) * np.array([-1, 0, 1])  # get s
                s = np.hstack([s, s, s])
                # get weights
                w1 = (5 / 9) ** 2
                w2 = (5 / 9) * (8 / 9)
                w3 = (8 / 9) ** 2
                w = np.array([w1, w2, w1, w2, w3, w2, w1, w2, w1])  # weights to all corresponding points
            else:
                raise Exception("A higher order integration than 9 nodes for 2D has not been implemented.")
            if self.elementtype[4] == "4":
                # calc all parameters for the X and Y functions (Q4)
                A = np.array([[1, -1, -1, 1], [1, 1, -1, -1], [1, 1, 1, 1], [1, -1, 1, -1]])
                ax = np.linalg.inv(A) @ np.transpose(x[0])
                ay = np.linalg.inv(A) @ np.transpose(x[1])
                # total Ki for Q4 element
                Ki = np.zeros([8, 8])
                for i in range(0, Nint):  # for all Gauss points (N in total)
                    # [J] Jacobi matrix (only Q4)
                    J = np.array(
                        [[ax[1] + ax[3] * t[i], ay[1] + ay[3] * t[i]], [ax[2] + ax[3] * s[i], ay[2] + ay[3] * s[i]]]
                    )
                    # make inverse of Jacobi
                    invJ = np.linalg.inv(J)
                    # [b] connects displacement derivatives (Q4)
                    b = np.array([[invJ, np.zeros([2, 2])], [np.zeros([2, 2]), invJ]])
                    # make [b] what it should actually look like
                    b = b.transpose(0, 2, 1, 3).reshape(4, 4)
                    # [h] as a temporary matrix
                    h = np.array(
                        [
                            [-1 / 4 * (1 - t[i]), 0, 1 / 4 * (1 - t[i]), 0, 1 / 4 * (1 + t[i]), 0, -1 / 4 * (1 + t[i])],
                            [-1 / 4 * (1 - s[i]), 0, -1 / 4 * (1 + s[i]), 0, 1 / 4 * (1 + s[i]), 0, 1 / 4 * (1 - s[i])],
                        ]
                    )
                    # assemble [c] differentiated shapefunctions (Q4)
                    c = np.vstack([np.hstack([h, np.zeros([2, 1])]), np.hstack([np.zeros([2, 1]), h])])
                    # [B] for all different s and t
                    Bi = a @ b @ c
                    Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * th * w[i]
                    Ki = Ki + Kii
            elif self.elementtype[4] == "8":
                # calc all parameters for X and Y (Q8)
                A = np.array(
                    [
                        [1, -1, -1, 1, 1, 1, -1, -1],
                        [1, 1, -1, -1, 1, 1, -1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, -1, 1, -1, 1, 1, 1, -1],
                        [1, 0, -1, 0, 0, 1, 0, 0],
                        [1, 1, 0, 0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 1, 0, 0],
                        [1, -1, 0, 0, 1, 0, 0, 0],
                    ]
                )
                ax = np.linalg.inv(A) @ np.transpose(x[0])
                ay = np.linalg.inv(A) @ np.transpose(x[1])
                # total Ki for Q8 element
                Ki = np.zeros([16, 16])
                for i in range(0, Nint):  # for all Gauss points (N in total)
                    # [J] Jacobi matrix for Q8
                    J = np.array(
                        [
                            [
                                ax[1] + ax[3] * t[i] + 2 * ax[4] * s[i] + 2 * ax[6] * s[i] * t[i] + ax[7] * t[i] ** 2,
                                ay[1] + ay[3] * t[i] + 2 * ay[4] * s[i] + 2 * ay[6] * s[i] * t[i] + ay[7] * t[i] ** 2,
                            ],
                            [
                                ax[2] + ax[3] * s[i] + 2 * ax[5] * t[i] + ax[6] * s[i] ** 2 + 2 * ax[7] * s[i] * t[i],
                                ay[2] + ay[3] * s[i] + 2 * ay[5] * t[i] + ay[6] * s[i] ** 2 + 2 * ay[7] * s[i] * t[i],
                            ],
                        ]
                    )
                    # make inverse of Jacobi
                    invJ = np.linalg.inv(J)
                    # [b] connects displacement derivatives (Q8)
                    b = np.array([[invJ, np.zeros([2, 2])], [np.zeros([2, 2]), invJ]])
                    # make [b] what it should actually look like
                    b = b.transpose(0, 2, 1, 3).reshape(4, 4)
                    # [h] as a temporary matrix
                    h = np.array(
                        [
                            [
                                -1 / 4 * (-1 + t[i]) * (2 * s[i] + t[i]),
                                0,
                                1 / 4 * (-1 + t[i]) * (t[i] - 2 * s[i]),
                                0,
                                1 / 4 * (1 + t[i]) * (2 * s[i] + t[i]),
                                0,
                                -1 / 4 * (1 + t[i]) * (t[i] - 2 * s[i]),
                                0,
                                s[i] * (-1 + t[i]),
                                0,
                                -1 / 2 * (1 + t[i]) * (-1 + t[i]),
                                0,
                                -s[i] * (1 + t[i]),
                                0,
                                1 / 2 * (1 + t[i]) * (-1 + t[i]),
                            ],
                            [
                                -1 / 4 * (-1 + s[i]) * (s[i] + 2 * t[i]),
                                0,
                                1 / 4 * (1 + s[i]) * (2 * t[i] - s[i]),
                                0,
                                1 / 4 * (1 + s[i]) * (s[i] + 2 * t[i]),
                                0,
                                -1 / 4 * (-1 + s[i]) * (2 * t[i] - s[i]),
                                0,
                                1 / 2 * (1 + s[i]) * (-1 + s[i]),
                                0,
                                -t[i] * (1 + s[i]),
                                0,
                                -1 / 2 * (1 + s[i]) * (-1 + s[i]),
                                0,
                                t[i] * (-1 + s[i]),
                            ],
                        ]
                    )
                    # assemble [c] differentiated shapefunctions (Q8)
                    c = np.vstack([np.hstack([h, np.zeros([2, 1])]), np.hstack([np.zeros([2, 1]), h])])
                    # [B] for all different s and t
                    Bi = a @ b @ c
                    Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * th * w[i]
                    Ki = Ki + Kii
        elif self.elementtype[0:5] == "Hexa8":  # 3D hexahedron element
            # Ei = elasticity matrix
            Ei = (
                self.E
                / ((1 + self.v) * (1 - 2 * self.v))
                * np.array(
                    [
                        [(1 - self.v), self.v, self.v, 0, 0, 0],
                        [self.v, (1 - self.v), self.v, 0, 0, 0],
                        [self.v, self.v, (1 - self.v), 0, 0, 0],
                        [0, 0, 0, (1 - 2 * self.v) / 2, 0, 0],
                        [0, 0, 0, 0, (1 - 2 * self.v) / 2, 0],
                        [0, 0, 0, 0, 0, (1 - 2 * self.v) / 2],
                    ]
                )
            )
            # [a] matrix that connects strain and displacement derivatives
            a = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                ]
            )
            if self.elementtype[4:6] in ("8N", "8"):
                Nint = 8
                # get all points
                z = 1 / np.sqrt(3) * np.array([1, 1, 1, 1, -1, -1, -1, -1])  # get z (local z)
                t = 1 / np.sqrt(3) * np.array([-1, -1, 1, 1])  # get t
                s = 1 / np.sqrt(3) * np.array([-1, 1, 1, -1])  # get s
                t = np.hstack([t, t])
                s = np.hstack([s, s])
                # get weights
                w = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # weights to all corresponding points
            else:  # higher order integration follow later
                raise Exception("Higher order integration not possible yet: WIP")
            # calc all parameters for the X, Y and Z functions
            A = np.array(
                [
                    [1, -1, -1, -1, 1, 1, 1, -1],
                    [1, -1, -1, 1, 1, -1, -1, 1],
                    [1, 1, -1, 1, -1, -1, 1, -1],
                    [1, 1, -1, -1, -1, 1, -1, 1],
                    [1, -1, 1, -1, -1, -1, 1, 1],
                    [1, -1, 1, 1, -1, 1, -1, -1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, -1, 1, -1, -1, -1],
                ]
            )
            ax = np.linalg.inv(A) @ np.transpose(x[0])
            ay = np.linalg.inv(A) @ np.transpose(x[1])
            az = np.linalg.inv(A) @ np.transpose(x[2])
            # total Ki for Hex8 element
            Ki = np.zeros([24, 24])
            for i in range(0, Nint):  # for all Gauss points (N in total)
                # [J] Jacobi matrix (only H8)
                J = np.array(
                    [
                        [
                            ax[1] + ax[4] * t[i] + ax[6] * z[i] + ax[7] * t[i] * z[i],
                            ay[1] + ay[4] * t[i] + ay[6] * z[i] + ay[7] * t[i] * z[i],
                            az[1] + az[4] * t[i] + az[6] * z[i] + az[7] * t[i] * z[i],
                        ],
                        [
                            ax[2] + ax[4] * s[i] + ax[5] * z[i] + ax[7] * s[i] * z[i],
                            ay[2] + ay[4] * s[i] + ay[5] * z[i] + ay[7] * s[i] * z[i],
                            az[2] + az[4] * s[i] + az[5] * z[i] + az[7] * s[i] * z[i],
                        ],
                        [
                            ax[3] + ax[5] * t[i] + ax[6] * s[i] + ax[7] * s[i] * t[i],
                            ay[3] + ay[5] * t[i] + ay[6] * s[i] + ay[7] * s[i] * t[i],
                            az[3] + az[5] * t[i] + az[6] * s[i] + az[7] * s[i] * t[i],
                        ],
                    ]
                )
                # make inverse of Jacobi
                invJ = np.linalg.inv(J)
                # [b] connects displacement derivatives (Hex8)
                b = np.array(
                    [
                        [invJ, np.zeros([3, 3]), np.zeros([3, 3])],
                        [np.zeros([3, 3]), invJ, np.zeros([3, 3])],
                        [np.zeros([3, 3]), np.zeros([3, 3]), invJ],
                    ]
                )
                # make [b] what it should actually look like
                b = b.transpose(0, 2, 1, 3).reshape(9, 9)
                # [h] as a temporary matrix
                h = (
                    1
                    / 8
                    * np.array(
                        [
                            [
                                -(1 - t[i]) * (1 - z[i]),
                                0,
                                0,
                                -(1 - t[i]) * (1 + z[i]),
                                0,
                                0,
                                (1 - t[i]) * (1 + z[i]),
                                0,
                                0,
                                (1 - t[i]) * (1 - z[i]),
                                0,
                                0,
                                -(1 + t[i]) * (1 - z[i]),
                                0,
                                0,
                                -(1 + t[i]) * (1 + z[i]),
                                0,
                                0,
                                (1 + t[i]) * (1 + z[i]),
                                0,
                                0,
                                (1 + t[i]) * (1 - z[i]),
                            ],
                            [
                                -(1 - s[i]) * (1 - z[i]),
                                0,
                                0,
                                -(1 - s[i]) * (1 + z[i]),
                                0,
                                0,
                                -(1 + s[i]) * (1 + z[i]),
                                0,
                                0,
                                -(1 + s[i]) * (1 - z[i]),
                                0,
                                0,
                                (1 - s[i]) * (1 - z[i]),
                                0,
                                0,
                                (1 - s[i]) * (1 + z[i]),
                                0,
                                0,
                                (1 + s[i]) * (1 + z[i]),
                                0,
                                0,
                                (1 + s[i]) * (1 - z[i]),
                            ],
                            [
                                -(1 - s[i]) * (1 - t[i]),
                                0,
                                0,
                                (1 - s[i]) * (1 - t[i]),
                                0,
                                0,
                                (1 + s[i]) * (1 - t[i]),
                                0,
                                0,
                                -(1 + s[i]) * (1 - t[i]),
                                0,
                                0,
                                -(1 - s[i]) * (1 + t[i]),
                                0,
                                0,
                                (1 - s[i]) * (1 + t[i]),
                                0,
                                0,
                                (1 + s[i]) * (1 + t[i]),
                                0,
                                0,
                                -(1 + s[i]) * (1 + t[i]),
                            ],
                        ]
                    )
                )
                # assemble [c] differentiated shapefunctions (Hex8)
                c = np.vstack(
                    [
                        np.hstack([h, np.zeros([3, 2])]),
                        np.hstack([np.zeros([3, 1]), h, np.zeros([3, 1])]),
                        np.hstack([np.zeros([3, 2]), h]),
                    ]
                )
                # [B] for all different s and t
                Bi = a @ b @ c
                Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * w[i]
                Ki = Ki + Kii
        K[:] = Ki  # add Ki
        self.K_ = K
    
    def computeStress(self, stress: np.array, dStressDDDeformationGradient: np.array, StrOld: np.array, StrNew: np.array, timeNew: float, dT: float):
        pass
