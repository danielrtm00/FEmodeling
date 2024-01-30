import numpy as np
import numpy.linalg as lin
from fe.points.node import Node
from fe.elements.base.baseelement import BaseElement
import pdb


class Element(BaseElement):
    @property
    def elNumber(self) -> int:
        """The unique number of this element"""

        return self.elnumber  # return number

    @property
    def nNodes(self) -> int:
        """The number of nodes this element requires"""

        if self.elementtype[0:5] == "Quad4":  # for 4 nodes
            return 4
        elif self.elementtype[0:5] in ("Quad8", "Hexa8"):  # for 8 nodes
            return 8
        else:
            raise Exception("This element Type doesn't exist")

    @property
    def nodes(self) -> int:
        """The list of nodes this element holds"""

        return self._nodes

    @property
    def nDof(self) -> int:
        """The total number of degrees of freedom this element has"""

        if self.elementtype[0:5] == "Quad4":  # for 4 nodes
            return 8
        elif self.elementtype[0:5] == "Quad8":  # for 8 nodes
            return 16
        elif self.elementtype[0:5] == "Hexa8":  # for hexahedron 3D with 8 nodes
            return 24
        else:
            raise Exception("This element Type doesn't exist")

    @property
    def fields(self) -> list[list[str]]:
        """The list of fields per nodes."""

        if self.elementtype[0:5] == "Quad4":
            return [["displacement"], ["displacement"], ["displacement"], ["displacement"]]
        elif self.elementtype[0:5] in ("Quad8", "Hexa8"):
            return [
                ["displacement"],
                ["displacement"],
                ["displacement"],
                ["displacement"],
                ["displacement"],
                ["displacement"],
                ["displacement"],
                ["displacement"],
            ]
        else:
            raise Exception("This element Type doesn't exist")

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness matrix to
        aggregate all entries in order to resemble the defined fields nodewise.
        In this case it stays the same because we use the nodes exactly like they are."""

        if self.elementtype[0:5] == "Quad4":  # for 4 nodes
            return np.arange(0, 8)
        elif self.elementtype[0:5] == "Quad8":  # for 8 nodes
            return np.arange(0, 16)
        elif self.elementtype[0:5] == "Hexa8":  # for hexahedron 3D with 8 nodes
            return np.arange(0, 24)
        else:
            raise Exception("This element Type doesn't exist")

    @property
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""

        if self.elementtype[0:5] == "Quad4":
            return "quad4"
        elif self.elementtype[0:5] == "Quad8":
            return "quad8"
        elif self.elementtype[0:5] == "Hexa8":
            return "hexa8"
        else:
            raise Exception("This element Type doesn't exist")

    def __init__(self, elementType: str, elNumber: int):
        """This element can be used for EdelweissFE. The element currently only allows calculations with node forces and given displacements.

        Parameters
        ----------
        elementType
            A string identifying the requested element formulation as shown below.
        elNumber
            A unique integer label used for all kinds of purposes."""

        """
        The following types of elements and attributes are currently possible (elementType):

        Elements
        --------
            Quad4
                quadrilateral element with 4 nodes.
            Quad8
                quadrilateral element with 8 nodes.
            Hexa8
                hexahedron element with 8 nodes.

        optional Parameters
        -------------------
        The following attributes are also included in the elementtype definition:
        
            R
                reduced integration for element, in elementtype[5].
            N
                regular/normal integration for element, in elementtype[5].
            E
                extended integration for element, in elementtype[5].
            PE
                use plane strain for 2D elements, in elementtype[6:8] or [5:7].
            PS
                use plane stress for 2D elements, in elementtype[6:8] or [5:7].

        If R, N or E is not given by the user, we assume N.
        If PE or PS is not given by the user, we assume PE."""

        self.elementtype = elementType[0].upper() + elementType[1:5].lower() + elementType[5:].upper()
        self.elnumber = elNumber

    def setNodes(self, nodes: list[Node]):
        """Assign the nodes to the element.

        Parameters
        ----------
        nodes
            A list of nodes.
        """

        self._nodes = nodes
        _nodesCoordinates = np.array([n.coordinates for n in nodes])  # get node coordinates
        self._nodesCoordinates = _nodesCoordinates.transpose()  # nodes given column-wise: x-coordinate - y-coordinate

    def setProperties(self, elementProperties: np.ndarray):
        """Assign a set of properties to the element.

        Parameters
        ----------
        elementProperties
            A numpy array containing the element properties.

        Attributes
        ----------
        thickness
            Thickness of 2D elements.
        """

        if self.elementtype[0] in ("T", "Q"):
            self.th = elementProperties[0]  # thickness

    def initializeElement(
        self,
    ):
        """Initalize the element to be ready for computing."""

        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Assign a material and material properties.

        Parameters
        ----------
        materialName
            The name of the requested material.
        materialProperties
            The numpy array containing the material properties.

        Attributes
        ----------
        E
            Elasticity module, has to be given first.
        v
            Poisson's ratio, has to be given second.
        """
        self.E = materialProperties[0]  # set E
        self.v = materialProperties[1]  # set v

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        """Assign initial conditions.

        Parameters
        ----------
        stateType
            The type of initial state.
        values
            The numpy array describing the initial state.
        """

        pass

    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: np.ndarray,
        dTime: float,
    ):
        """Evaluate residual and stiffness for given time, field, and field increment due to a surface load.

        Parameters
        ----------
        loadType
            The type of load.
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        faceID
            The number of the elements face this load acts on.
        load
            The magnitude (or vector) describing the load.
        U
            The current solution vector.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """

        pass

    def computeYourself(
        self,
        K_: np.ndarray,
        P: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: np.ndarray,
        dTime: float,
    ):
        """Evaluate the residual and stiffness matrix for given time, field, and field increment due to a displacement or load.

        Parameters
        ----------
        P
            The external load vector gets calculated.
        K
            The stiffness matrix gets calculated.
        U
            The current solution vector.
        dU
            The current solution vector increment.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """

        # assume it's plain strain if it's not given by user
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
        Ndof = self.nDof
        K = np.reshape(K_, (Ndof, Ndof))
        x = self._nodesCoordinates
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
                    Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * self.th * w[i]
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
                    Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * self.th * w[i]
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
        P[:] -= Ki @ U

    def computeBodyForce(
        self, P: np.ndarray, K: np.ndarray, load: np.ndarray, U: np.ndarray, time: np.ndarray, dTime: float
    ):
        """Evaluate residual and stiffness for given time, field, and field increment due to a body force load.

        Parameters
        ----------
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        load
            The magnitude (or vector) describing the load.
        U
            The current solution vector.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """

        pass

    def acceptLastState(
        self,
    ):
        """Accept the computed state (in nonlinear iteration schemes)."""

        pass

    def resetToLastValidState(
        self,
    ):
        """Rest to the last valid state."""

        pass

    def getResultArray(self, result: str, quadraturePoint: int, getPersistentView: bool = True) -> np.ndarray:
        """Get the array of a result, possibly as a persistent view which is continiously
        updated by the element.

        Parameters
        ----------
        result
            The name of the result.
        quadraturePoint
            The number of the quadrature point.
        getPersistentView
            If true, the returned array should be continiously updated by the element.

        Returns
        -------
        np.ndarray
            The result.
        """

        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Compute the underlying MarmotElement centroid coordinates.

        Returns
        -------
        np.ndarray
            The element's central coordinates.
        """

        x=self._nodesCoordinates
        return np.average(x, axis=1)

    def getNumberOfQuadraturePoints(self) -> int:
        """Compute the underlying MarmotElement qp coordinates.

        Returns
        -------
        np.ndarray
            The element's qp coordinates.
        """
        
        if self.elementtype[0] == "Q":  # it's a 2D quadrilateral element
            if self.elementtype[4:6] in ("4N", "8R", "4") or (
                len(self.elementtype) == 7 and self.elementtype[4] == "4"
            ):  return 4
            elif self.elementtype[4:6] == "4R":
                return 1
            elif self.elementtype[4:6] in ("8N", "4E", "8") or (
                len(self.elementtype) == 7 and self.elementtype[4] == "8"
            ):  return 9
            else:
                raise Exception("Higher order integration not possible yet: WIP")
        elif self.elementtype[0:5] == "Hexa8":
            if self.elementtype[4:6] in ("8N", "8"):
                return 8
            else:
                raise Exception("Higher order integration not possible yet: WIP")
        else:
            raise Exception("This element Type doesn't exist.")

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        """Compute the underlying MarmotElement qp coordinates.

        Returns
        -------
        np.ndarray
            The element's qp coordinates.
        """

        return None
