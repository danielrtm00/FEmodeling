import numpy as np
import numpy.linalg as lin
from fe.points.node import Node
from fe.elements.base.baseelement import BaseElement
from fe.materials.EdelweissMaterial import EdelweissMaterial
import pdb


class Element(BaseElement):
    @property
    def elNumber(self) -> int:
        """The unique number of this element"""

        return self._elNumber  # return number

    @property
    def nNodes(self) -> int:
        """The number of nodes this element requires"""

        return self.diction["nNodes"]

    @property
    def nodes(self) -> int:
        """The list of nodes this element holds"""

        return self._nodes

    @property
    def nDof(self) -> int:
        """The total number of degrees of freedom this element has"""

        return self.diction["nDof"]

    @property
    def fields(self) -> list[list[str]]:
        """The list of fields per nodes."""

        return self.diction["fields"]

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness matrix to
        aggregate all entries in order to resemble the defined fields nodewise.
        In this case it stays the same because we use the nodes exactly like they are."""

        return self.diction["dofIndices"]

    @property
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""

        return self.diction["ensightType"]

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
        self._elNumber = elNumber
        if self.elementtype[0] == "Q":
            if self.elementtype[4] == "4":
                self.diction = {"nNodes": 4, "nDof": 8, "dofIndices": np.arange(0, 8), "ensightType": "quad4", "dim": 2}
            elif self.elementtype[4] == "8":
                self.diction = {
                    "nNodes": 8,
                    "nDof": 16,
                    "dofIndices": np.arange(0, 16),
                    "ensightType": "quad8",
                    "dim": 2,
                }
            else:
                raise Exception("Elements of higher order than quadratical have not been implemented.")
            if self.elementtype[4:6] in ("4N", "8R", "4") or (
                len(self.elementtype) == 7 and self.elementtype[4] == "4"
            ):
                self.diction.update(
                    {
                        "Nint": 4,
                        "t": 1 / np.sqrt(3) * np.array([-1, -1, 1, 1]),
                        "s": 1 / np.sqrt(3) * np.array([-1, 1, 1, -1]),
                        "w": np.ones(4),
                    }
                )
            elif self.elementtype[4:6] == "4R":
                self.diction.update({"Nint": 1, "t": np.array([0]), "s": np.array([0]), "w": np.array([4])})
            elif self.elementtype[4:6] in ("8N", "4E", "8") or (
                len(self.elementtype) == 7 and self.elementtype[4] == "8"
            ):
                s = np.sqrt(0.6) * np.array([-1, 0, 1])
                w1 = (5 / 9) ** 2
                w2 = (5 / 9) * (8 / 9)
                w3 = (8 / 9) ** 2
                self.diction.update(
                    {
                        "Nint": 9,
                        "t": np.sqrt(0.6) * np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1]),
                        "s": np.hstack([s, s, s]),
                        "w": np.array([w1, w2, w1, w2, w3, w2, w1, w2, w1]),
                    }
                )
            else:
                raise Exception("A higher order integration than 9 nodes for 2D has not been implemented.")
            if self.elementtype[6:8] in ("PE", "") or (len(self.elementtype) == 7 and self.elementtype[5:7] == "PE"):
                self.diction.update({"plStrain": True})
            elif self.elementtype[6:8] == "PS" or (len(self.elementtype) == 7 and self.elementtype[5:7] == "PS"):
                self.diction.update({"plStrain": False})
        elif self.elementtype[0] == "H":
            if self.elementtype[4] == "8":
                self.diction = {
                    "nNodes": 8,
                    "nDof": 24,
                    "dofIndices": np.arange(0, 24),
                    "ensightType": "hexa8",
                    "dim": 3,
                }
            elif self.elementtype[4:6] == "20":
                self.diction = {
                    "nNodes": 20,
                    "nDof": 60,
                    "dofIndices": np.arange(0, 60),
                    "ensightType": "hexa20",
                    "dim": 3,
                }
            else:
                raise Exception("Elements of higher order than quadratical have not been implemented.")
            if self.elementtype[4:7] in ("8N", "8", "20R"):
                t = 1 / np.sqrt(3) * np.array([-1, -1, 1, 1])  # get t
                s = 1 / np.sqrt(3) * np.array([-1, 1, 1, -1])  # get s
                self.diction.update(
                    {
                        "plStrain": False,
                        "Nint": 8,
                        "z": 1 / np.sqrt(3) * np.array([1, 1, 1, 1, -1, -1, -1, -1]),
                        "t": np.hstack([t, t]),
                        "s": np.hstack([s, s]),
                        "w": np.ones(8),
                    }
                )
            elif self.elementtype[4:7] in ("20N", "20"):
                z = np.sqrt(0.6) * np.array([-1, 0, 1]) # get part of z
                s = np.sqrt(0.6) * np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])  # get part of s
                w1 = (5 / 9) ** 3
                w2 = (5 / 9) ** 2 * (8/9)
                w3 = (5/9) * (8/9) ** 2
                w4 = (8/9) ** 3
                w = np.array([w1, w2, w1, w2, w3, w2, w1, w2, w1, w2, w3, w2, w3])
                self.diction.update(
                    {
                        "plStrain": False,
                        "Nint": 27,
                        "z": np.hstack([z, z, z, z, z, z, z, z, z]),
                        "t": np.sqrt(0.6) * np.hstack([-np.ones(9), np.zeros(9), np.ones(9)]),
                        "s": np.hstack([s, s, s]),
                        "w": np.hstack([w, w4, w[::-1]]),
                    }
                )
            elif self.elementtype[4:6] == "8R":
                self.diction.update({"plStrain": False, "Nint": 1, "z": np.array([0]), "t": np.array([0]), "s": np.array([0]), "w": np.array([4])})
            else:
                raise Exception("A higher order integration than 20 nodes for 3D has not been implemented.")
        else:
            raise Exception("This element Type doesn't exist.")
        self.diction.update({"fields": [["displacement"] for i in range(self.diction["nNodes"])]})

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

        if self.elementtype[0] == "Q":
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
        self.materialname = materialName  # set material
        self.material = EdelweissMaterial(self.materialname, materialProperties)

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
        Ndof = self.nDof
        K = np.reshape(K_, (Ndof, Ndof))
        x = self._nodesCoordinates
        Nint = self.diction["Nint"]
        t = self.diction["t"]
        s = self.diction["s"]
        w = self.diction["w"]
        dim = self.diction["dim"]
        J = np.zeros([Nint, dim, dim])
        b = np.zeros([Nint, dim * dim, dim * dim])
        c = np.zeros([Nint, dim * dim, Ndof])
        if dim == 2:  # it's a 2D quadrilateral element
            # [a] matrix that connects strain and displacement derivatives
            a = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
            # integration points
            stress = np.zeros([Nint, 3])
            dStrain = np.zeros([Nint, 3])
            dStressdStrain = np.zeros([3, 3])
            if self.diction["nNodes"] == 4:
                # calc all parameters for the X and Y functions (Q4)
                A = np.array([[1, -1, -1, 1], [1, 1, -1, -1], [1, 1, 1, 1], [1, -1, 1, -1]])
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                # total Ki for Q4 element
                Ki = np.zeros([8, 8])
                for i in range(0, Nint):  # for all Gauss points (N in total)
                    # [J] Jacobi matrix (only Q4)
                    J[:][:][i] = np.array(
                        [[ax[1] + ax[3] * t[i], ay[1] + ay[3] * t[i]], [ax[2] + ax[3] * s[i], ay[2] + ay[3] * s[i]]]
                    )
                    # make inverse of Jacobi
                    invJ = np.linalg.inv(J[:][:][i])
                    # [b] connects displacement derivatives (Q4)
                    bi = np.array([[invJ, np.zeros([2, 2])], [np.zeros([2, 2]), invJ]])
                    # make [b] what it should actually look like
                    b[:][:][i] = bi.transpose(0, 2, 1, 3).reshape(4, 4)
                    # [h] as a temporary matrix
                    h = np.array(
                        [
                            [-1 / 4 * (1 - t[i]), 0, 1 / 4 * (1 - t[i]), 0, 1 / 4 * (1 + t[i]), 0, -1 / 4 * (1 + t[i])],
                            [-1 / 4 * (1 - s[i]), 0, -1 / 4 * (1 + s[i]), 0, 1 / 4 * (1 + s[i]), 0, 1 / 4 * (1 - s[i])],
                        ]
                    )
                    # assemble [c] differentiated shapefunctions (Q4)
                    c[:][:][i] = np.vstack([np.hstack([h, np.zeros([2, 1])]), np.hstack([np.zeros([2, 1]), h])])
            elif self.diction["nNodes"] == 8:
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
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                # total Ki for Q8 element
                Ki = np.zeros([16, 16])
                for i in range(0, Nint):  # for all Gauss points (N in total)
                    # [J] Jacobi matrix for Q8
                    J[:][:][i] = np.array(
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
                    invJ = np.linalg.inv(J[:][:][i])
                    # [b] connects displacement derivatives (Q8)
                    bi = np.array([[invJ, np.zeros([2, 2])], [np.zeros([2, 2]), invJ]])
                    # make [b] what it should actually look like
                    b[:][:][i] = bi.transpose(0, 2, 1, 3).reshape(4, 4)
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
                    c[:][:][i] = np.vstack([np.hstack([h, np.zeros([2, 1])]), np.hstack([np.zeros([2, 1]), h])])
        elif dim == 3:  # 3D hexahedron element
            z = self.diction["z"]
            stress = np.zeros([Nint, 6])
            dStrain = np.zeros([Nint, 6])
            dStressdStrain = np.zeros([6, 6])
            self.th = 1  # set thickness to 1 (no thickness for 3D elements)
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
            if self.diction["nNodes"] == 8:
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
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                az = invA @ np.transpose(x[2])
                # total Ki for Hex8 element
                Ki = np.zeros([24, 24])
                for i in range(0, Nint):  # for all Gauss points (N in total)
                    # [J] Jacobi matrix (only H8)
                    J[:][:][i] = np.array(
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
                    invJ = np.linalg.inv(J[:][:][i])
                    # [b] connects displacement derivatives (Hex8)
                    bi = np.array(
                        [
                            [invJ, np.zeros([3, 3]), np.zeros([3, 3])],
                            [np.zeros([3, 3]), invJ, np.zeros([3, 3])],
                            [np.zeros([3, 3]), np.zeros([3, 3]), invJ],
                        ]
                    )
                    # make [b] what it should actually look like
                    b[:][:][i] = bi.transpose(0, 2, 1, 3).reshape(9, 9)
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
                    # assemble [c] differentiated shapefunctions (Hexa8)
                    c[:][:][i] = np.vstack(
                        [
                            np.hstack([h, np.zeros([3, 2])]),
                            np.hstack([np.zeros([3, 1]), h, np.zeros([3, 1])]),
                            np.hstack([np.zeros([3, 2]), h]),
                        ]
                    )
            if self.diction["nNodes"] == 20:
                raise Exception("This function is not possible for 20 nodes (yet).")
                A = np.array([[ 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1],
                              [ 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1],
                              [ 1, 1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1],
                              [ 1, 1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1],
                              [ 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1],
                              [ 1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1],
                              [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [ 1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1],
                              [ 1,-1,-1, 0, 1, 0, 0, 1, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0,-1, 1, 0,-1, 0, 0, 1, 1, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0],
                              [ 1, 1,-1, 0,-1, 0, 0, 1, 1, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0,-1,-1, 0, 1, 0, 0, 1, 1, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0],
                              [ 1,-1, 1, 0,-1, 0, 0, 1, 1, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                              [ 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0, 1,-1, 0,-1, 0, 0, 1, 1, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
                              [ 1,-1, 0,-1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0],
                              [ 1,-1, 0, 1, 0, 0,-1, 1, 0, 1, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0],
                              [ 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [ 1, 1, 0,-1, 0, 0,-1, 1, 0, 1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0]])
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                az = invA @ np.transpose(x[2])
                # total Ki for Hex8 element
                Ki = np.zeros([60, 60])
                for i in range(0, Nint):  # for all Gauss points (N in total)
                    # [J] Jacobi matrix (only H8)
                    J[:][:][i] = np.array(
                        [
                            [
                                ax[1]+ax[4]*t[i]+ax[6]*z[i]+2*ax[7]*s[i]+2*ax[10]*s[i]*t[i]+ax[11]*t[i]**2+ax[14]*z[i]**2+2*ax[15]*z[i]*s[i]+ax[16]*t[i]*z[i]+2*ax[17]*s[i]*t[i]*z[i]+ax[18]*t[i]**2*z[i]+ax[19]*t[i]*z[i]**2,
                                ay[1]+ay[4]*t[i]+ay[6]*z[i]+2*ay[7]*s[i]+2*ay[10]*s[i]*t[i]+ay[11]*t[i]**2+ay[14]*z[i]**2+2*ay[15]*z[i]*s[i]+ay[16]*t[i]*z[i]+2*ay[17]*s[i]*t[i]*z[i]+ay[18]*t[i]**2*z[i]+ay[19]*t[i]*z[i]**2,
                                az[1]+az[4]*t[i]+az[6]*z[i]+2*az[7]*s[i]+2*az[10]*s[i]*t[i]+az[11]*t[i]**2+az[14]*z[i]**2+2*az[15]*z[i]*s[i]+az[16]*t[i]*z[i]+2*az[17]*s[i]*t[i]*z[i]+az[18]*t[i]**2*z[i]+az[19]*t[i]*z[i]**2,
                            ],
                            [
                                ax[2]+ax[4]*s[i]+ax[5]*z[i]+2*ax[8]*t[i]+ax[10]*s[i]**2+2*ax[11]*s[i]*t[i]+2*ax[12]*t[i]*z[i]+ax[13]*z[i]**2+ax[16]*s[i]*z[i]+ax[17]*s[i]**2*z[i]+2*ax[18]*s[i]*t[i]*z[i]+ax[19]*s[i]*z[i]**2,
                                ay[2]+ay[4]*s[i]+ay[5]*z[i]+2*ay[8]*t[i]+ay[10]*s[i]**2+2*ay[11]*s[i]*t[i]+2*ay[12]*t[i]*z[i]+ay[13]*z[i]**2+ay[16]*s[i]*z[i]+ay[17]*s[i]**2*z[i]+2*ay[18]*s[i]*t[i]*z[i]+ay[19]*s[i]*z[i]**2,
                                az[2]+az[4]*s[i]+az[5]*z[i]+2*az[8]*t[i]+az[10]*s[i]**2+2*az[11]*s[i]*t[i]+2*az[12]*t[i]*z[i]+az[13]*z[i]**2+az[16]*s[i]*z[i]+az[17]*s[i]**2*z[i]+2*az[18]*s[i]*t[i]*z[i]+az[19]*s[i]*z[i]**2,
                            ],
                            [
                                ax[3]+ax[5]*z[i]+ax[6]*s[i]+2*ax[9]*z[i]+ax[12]*t[i]**2+2*ax[13]*t[i]*z[i]+2*ax[14]*z[i]*s[i]+ax[15]*s[i]**2+ax[16]*s[i]*t[i]+ax[17]*s[i]**2*t[i]+ax[18]*s[i]*t[i]**2+2*ax[19]*s[i]*t[i]*z[i],
                                ay[3]+ay[5]*z[i]+ay[6]*s[i]+2*ay[9]*z[i]+ay[12]*t[i]**2+2*ay[13]*t[i]*z[i]+2*ay[14]*z[i]*s[i]+ay[15]*s[i]**2+ay[16]*s[i]*t[i]+ay[17]*s[i]**2*t[i]+ay[18]*s[i]*t[i]**2+2*ay[19]*s[i]*t[i]*z[i],
                                az[3]+az[5]*z[i]+az[6]*s[i]+2*az[9]*z[i]+az[12]*t[i]**2+2*az[13]*t[i]*z[i]+2*az[14]*z[i]*s[i]+az[15]*s[i]**2+az[16]*s[i]*t[i]+az[17]*s[i]**2*t[i]+az[18]*s[i]*t[i]**2+2*az[19]*s[i]*t[i]*z[i],
                            ],
                        ]
                    )
                    # make inverse of Jacobi
                    invJ = np.linalg.inv(J[:][:][i])
                    # make [b] - connects displacement derivatives (Hexa8)
                    b[:][:][i] = np.bmat([[invJ, np.zeros([3, 3]), np.zeros([3, 3])],
                                          [np.zeros([3, 3]), invJ, np.zeros([3, 3])],
                                          [np.zeros([3, 3]), np.zeros([3, 3]), invJ]])
                    # [h] as a temporary matrix
                    h = np.array(
                        [
                            [
                                ((t[i] - 1)*(z[i] - 1)*(2*s[i] + t[i] + z[i] + 1))/8,
                                0,
                                0,
                                -((t[i] - 1)*(z[i] + 1)*(2*s[i] + t[i] - z[i] + 1))/8,
                                0,
                                0,
                                -((t[i] - 1)*(z[i] + 1)*(2*s[i] - t[i] + z[i] - 1))/8,
                                0,
                                0,
                                -((t[i] - 1)*(z[i] - 1)*(t[i] - 2*s[i] + z[i] + 1))/8,
                                0,
                                0,
                                -((t[i] + 1)*(z[i] - 1)*(2*s[i] - t[i] + z[i] + 1))/8,
                                0,
                                0,
                                ((t[i] + 1)*(z[i] + 1)*(2*s[i] - t[i] - z[i] + 1))/8,
                                0,
                                0,
                                ((t[i] + 1)*(z[i] + 1)*(2*s[i] + t[i] + z[i] - 1))/8,
                                0,
                                0,
                                -((t[i] + 1)*(z[i] - 1)*(2*s[i] + t[i] - z[i] - 1))/8,
                                0,
                                0,
                                -((z[i]**2 - 1)*(t[i] - 1))/4,
                                0,
                                0,
                                (s[i]*(t[i] - 1)*(z[i] + 1))/2,
                                0,
                                0,
                                ((z[i]**2 - 1)*(t[i] - 1))/4,
                                0,
                                0,
                                -(s[i]*(t[i] - 1)*(z[i] - 1))/2,
                                0,
                                0,
                                ((z[i]**2 - 1)*(t[i] + 1))/4,
                                0,
                                0,
                                -(s[i]*(t[i] + 1)*(z[i] + 1))/2,
                                0,
                                0,
                                -((z[i]**2 - 1)*(t[i] + 1))/4,
                                0,
                                0,
                                (s[i]*(t[i] + 1)*(z[i] - 1))/2,
                                0,
                                0,
                                -((t[i]**2 - 1)*(z[i] - 1))/4,
                                0,
                                0,
                                ((t[i]**2 - 1)*(z[i] + 1))/4,
                                0,
                                0,
                                -((t[i]**2 - 1)*(z[i] + 1))/4,
                                0,
                                0,
                                ((t[i]**2 - 1)*(z[i] - 1))/4,
                            ],
                            [
                                ((s[i] - 1)*(z[i] - 1)*(s[i] + 2*t[i] + z[i] + 1))/8,
                                0,
                                0,
                                -((s[i] - 1)*(z[i] + 1)*(s[i] + 2*t[i] - z[i] + 1))/8,
                                0,
                                0,
                                -((s[i] + 1)*(z[i] + 1)*(s[i] - 2*t[i] + z[i] - 1))/8,
                                0,
                                0,
                                -((s[i] + 1)*(z[i] - 1)*(2*t[i] - s[i] + z[i] + 1))/8,
                                0,
                                0,
                                -((s[i] - 1)*(z[i] - 1)*(s[i] - 2*t[i] + z[i] + 1))/8,
                                0,
                                0,
                                ((s[i] - 1)*(z[i] + 1)*(s[i] - 2*t[i] - z[i] + 1))/8,
                                0,
                                0,
                                ((s[i] + 1)*(z[i] + 1)*(s[i] + 2*t[i] + z[i] - 1))/8,
                                0,
                                0,
                                -((s[i] + 1)*(z[i] - 1)*(s[i] + 2*t[i] - z[i] - 1))/8,
                                0,
                                0,
                                -(s[i]/4 - 1/4)*(z[i]**2 - 1),
                                0,
                                0,
                                (s[i]**2/4 - 1/4)*(z[i] + 1),
                                0,
                                0,
                                (s[i]/4 + 1/4)*(z[i]**2 - 1),
                                0,
                                0,
                                -(s[i]**2/4 - 1/4)*(z[i] - 1),
                                0,
                                0,
                                (s[i]/4 - 1/4)*(z[i]**2 - 1),
                                0,
                                0,
                                -(s[i]**2/4 - 1/4)*(z[i] + 1),
                                0,
                                0,
                                -(s[i]/4 + 1/4)*(z[i]**2 - 1),
                                0,
                                0,
                                (s[i]**2/4 - 1/4)*(z[i] - 1),
                                0,
                                0,
                                -2*t[i]*(s[i]/4 - 1/4)*(z[i] - 1),
                                0,
                                0,
                                2*t[i]*(s[i]/4 - 1/4)*(z[i] + 1),
                                0,
                                0,
                                -2*t[i]*(s[i]/4 + 1/4)*(z[i] + 1),
                                0,
                                0,
                                2*t[i]*(s[i]/4 + 1/4)*(z[i] - 1),
                            ],
                            [
                                ((s[i] - 1)*(t[i] - 1)*(s[i] + t[i] + 2*z[i] + 1))/8,
                                0,
                                0,
                                -((s[i] - 1)*(t[i] - 1)*(s[i] + t[i] - 2*z[i] + 1))/8,
                                0,
                                0,
                                -((s[i] + 1)*(t[i] - 1)*(s[i] - t[i] + 2*z[i] - 1))/8,
                                0,
                                0,
                                -((s[i] + 1)*(t[i] - 1)*(t[i] - s[i] + 2*z[i] + 1))/8,
                                0,
                                0,
                                -((s[i] - 1)*(t[i] + 1)*(s[i] - t[i] + 2*z[i] + 1))/8,
                                0,
                                0,
                                ((s[i] - 1)*(t[i] + 1)*(s[i] - t[i] - 2*z[i] + 1))/8,
                                0,
                                0,
                                ((s[i] + 1)*(t[i] + 1)*(s[i] + t[i] + 2*z[i] - 1))/8,
                                0,
                                0,
                                -((s[i] + 1)*(t[i] + 1)*(s[i] + t[i] - 2*z[i] - 1))/8,
                                0,
                                0,
                                -2*z[i]*(s[i]/4 - 1/4)*(t[i] - 1),
                                0,
                                0,
                                (s[i]**2/4 - 1/4)*(t[i] - 1),
                                0,
                                0,
                                2*z[i]*(s[i]/4 + 1/4)*(t[i] - 1),
                                0,
                                0,
                                -(s[i]**2/4 - 1/4)*(t[i] - 1),
                                0,
                                0,
                                2*z[i]*(s[i]/4 - 1/4)*(t[i] + 1),
                                0,
                                0,
                                -(s[i]**2/4 - 1/4)*(t[i] + 1),
                                0,
                                0,
                                -2*z[i]*(s[i]/4 + 1/4)*(t[i] + 1),
                                0,
                                0,
                                (s[i]**2/4 - 1/4)*(t[i] + 1),
                                0,
                                0,
                                -(s[i]/4 - 1/4)*(t[i]**2 - 1),
                                0,
                                0,
                                (s[i]/4 - 1/4)*(t[i]**2 - 1),
                                0,
                                0,
                                -(s[i]/4 + 1/4)*(t[i]**2 - 1),
                                0,
                                0,
                                (s[i]/4 + 1/4)*(t[i]**2 - 1),
                            ]
                        ]
                    )
                    # assemble [c] differentiated shapefunctions (Hexa20)
                    c[:][:][i] = np.vstack(
                        [
                            np.hstack([h, np.zeros([3, 2])]),
                            np.hstack([np.zeros([3, 1]), h, np.zeros([3, 1])]),
                            np.hstack([np.zeros([3, 2]), h]),
                        ]
                    )
        plStrain = self.diction["plStrain"]
        for i in range(0, Nint):
            # [B] for all different s and t
            Bi = a @ b[:][:][i] @ c[:][:][i]
            # get stress and strain
            dStrain[i] = Bi @ dU
            dic = self.material.computeStress(stress[i], dStressdStrain, dStrain[i], time, dTime, dim, plStrain)
            stress[i] = np.concatenate(self.material.stress, axis=None)
            dStressdStrain = self.material.dStressdStrain
            # get stiffness matrix for element j in point i
            Kii = np.transpose(Bi) @ dStressdStrain @ Bi * lin.det(J[:][:][i]) * self.th * w[i]
            # calculate P
            P -= np.transpose(Bi) @ stress[i] * lin.det(J[:][:][i]) * w[i] * self.th
            Ki += Kii
        K[:] = Ki  # add Ki

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

        x = self._nodesCoordinates
        return np.average(x, axis=1)

    def getNumberOfQuadraturePoints(self) -> int:
        """Compute the underlying MarmotElement qp coordinates.

        Returns
        -------
        np.ndarray
            The element's qp coordinates.
        """

        return self.diction["Nint"]

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        """Compute the underlying MarmotElement qp coordinates.

        Returns
        -------
        np.ndarray
            The element's qp coordinates.
        """
        x = self._nodesCoordinates
        t = self.diction["t"]
        s = self.diction["s"]
        Nint = self.diction["Nint"]
        dim = self.diction["dim"]
        xI = np.zeros([Nint, dim])
        if dim == 2:  # it's a 2D quadrilateral element
            if self.diction["nNodes"] == 4:
                A = np.array([[1, -1, -1, 1], [1, 1, -1, -1], [1, 1, 1, 1], [1, -1, 1, -1]])
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                for i in range(0, Nint):
                    xI[i][0] = ax[0] + ax[1] * s[i] + ax[2] * t[i] + ax[3] * t[i] * s[i]
                    xI[i][1] = ay[0] + ay[1] * s[i] + ay[2] * t[i] + ay[3] * t[i] * s[i]
            elif self.diction["nNodes"] == 8:
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
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                for i in range(0, Nint):
                    xI[i][0] = (
                        ax[0]
                        + ax[1] * s[i]
                        + ax[2] * t[i]
                        + ax[3] * t[i] * s[i]
                        + ax[4] * s[i] ** 2
                        + ax[5] * t[i] ** 2
                        + ax[6] * s[i] ** 2 * t[i]
                        + ax[8] * s[i] * t[i] ** 2
                    )
                    xI[i][1] = (
                        ay[0]
                        + ay[1] * s[i]
                        + ay[2] * t[i]
                        + ay[3] * t[i] * s[i]
                        + ay[4] * s[i] ** 2
                        + ay[5] * t[i] ** 2
                        + ay[6] * s[i] ** 2 * t[i]
                        + ay[8] * s[i] * t[i] ** 2
                    )
        elif dim == 3:  # 3D hexahedron element
            z = self.diction["z"]
            if self.diction["nNodes"] == 8:
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
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                az = invA @ np.transpose(x[2])
                for i in range(0, Nint):
                    xI[i][0] = (
                        ax[0]
                        + ax[1] * s[i]
                        + ax[2] * t[i]
                        + ax[3] * z[i]
                        + ax[4] * s[i] * t[i]
                        + ax[5] * t[i] * z[i]
                        + ax[6] * s[i] * z[i]
                        + ax[7] * s[i] * t[i] * z[i]
                    )
                    xI[i][1] = (
                        ay[0]
                        + ay[1] * s[i]
                        + ay[2] * t[i]
                        + ay[3] * z[i]
                        + ay[4] * s[i] * t[i]
                        + ay[5] * t[i] * z[i]
                        + ay[6] * s[i] * z[i]
                        + ay[7] * s[i] * t[i] * z[i]
                    )
                    xI[i][2] = (
                        az[0]
                        + az[1] * s[i]
                        + az[2] * t[i]
                        + az[3] * z[i]
                        + az[4] * s[i] * t[i]
                        + az[5] * t[i] * z[i]
                        + az[6] * s[i] * z[i]
                        + az[7] * s[i] * t[i] * z[i]
                    )
            elif self.diction["nNodes"] == 20:
                raise Exception("This function is not possible for 20 nodes (yet).")
                A = np.array([[ 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1],
                              [ 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1],
                              [ 1, 1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1],
                              [ 1, 1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1],
                              [ 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1],
                              [ 1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1],
                              [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [ 1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1],
                              [ 1,-1,-1, 0, 1, 0, 0, 1, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0,-1, 1, 0,-1, 0, 0, 1, 1, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0],
                              [ 1, 1,-1, 0,-1, 0, 0, 1, 1, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0,-1,-1, 0, 1, 0, 0, 1, 1, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0],
                              [ 1,-1, 1, 0,-1, 0, 0, 1, 1, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                              [ 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [ 1, 0, 1,-1, 0,-1, 0, 0, 1, 1, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
                              [ 1,-1, 0,-1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0],
                              [ 1,-1, 0, 1, 0, 0,-1, 1, 0, 1, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0],
                              [ 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [ 1, 1, 0,-1, 0, 0,-1, 1, 0, 1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0]])
                invA = np.linalg.inv(A)
                ax = invA @ np.transpose(x[0])
                ay = invA @ np.transpose(x[1])
                az = invA @ np.transpose(x[2])
                for i in range(0, Nint):
                    xI[i][0] = None
                    xI[i][1] = None
                    xI[i][2] = None
        return xI
