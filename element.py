import numpy as np
import numpy.linalg as lin
from fe.points.node import Node
from fe.elements.base.baseelement import BaseElement
import pdb

class Element(BaseElement):
    @property
    def elNumber(self) -> int:
        """The unique number of this element"""

        return self.elnumber # return number

    @property
    def nNodes(self) -> int:
        """The number of nodes this element requires"""

        if self.elementtype[0:2] == "Q4": # for 4 nodes
            return 4
        elif self.elementtype[0:2] in ("Q8","H8"): # for 8 nodes
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
   
        if self.elementtype[0:2] == "Q4": # for 4 nodes
            return 8
        elif self.elementtype[0:2] =="Q8": # for 8 nodes
            return 16
        elif self.elementtype[0:2] =="H8": # for hexahedron 3D with 8 nodes
            return 24
        else:
            raise Exception("This element Type doesn't exist")

    @property
    
    def fields(self) -> list[list[str]]:
        """The list of fields per nodes."""

        if self.elementtype[0:2] == "Q4":
            return [["displacement"],["displacement"],["displacement"],["displacement"]]
        elif self.elementtype[0:2] in ("Q8","H8"):
            return [["displacement"],["displacement"],["displacement"],["displacement"],["displacement"],["displacement"],["displacement"],["displacement"]]
        else:
            raise Exception("This element Type doesn't exist")

    @property
    
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness matrix to
        aggregate all entries in order to resemble the defined fields nodewise.
        In this case it stays the same because we use the nodes exactly like they are."""

        if self.elementtype[0:2] == "Q4": # for 4 nodes
            return np.arange(0,8)
        elif self.elementtype[0:2] =="Q8": # for 8 nodes
            return np.arange(0,16)
        elif self.elementtype[0:2] == "H8": # for hexahedron 3D with 8 nodes
            return np.arange(0,24)
        else:
            raise Exception("This element Type doesn't exist")

    @property
    
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""
        
        if self.elementtype[0:2] == "Q4":
            return "quad4"
        elif self.elementtype[0:2] == "Q8":
            return "quad8"
        elif self.elementtype[0:2] == "H8":
            return "hexa8"
        else:
            raise Exception("This element Type doesn't exist")
        

    
    def __init__(self, elementType: str, elNumber: int):
        """Finite elements in EdelweissFE should be derived from this
        base class in order to follow the general interface.

        EdelweissFE expects the layout of the internal and external load vectors, P, PExt, (and the stiffness)
        to be of the form

        .. code-block:: console

            [ node 1 - dofs field 1,
              node 1 - dofs field 2,
              node 1 - ... ,
              node 1 - dofs field n,
              node 2 - dofs field 1,
              ... ,
              node N - dofs field n].

        Parameters
        ----------
        elementType
            A string identifying the requested element formulation.
        elNumber
            A unique integer label used for all kinds of purposes.
        
        The following types of elements and attributes are currently possible (elementType):

        2D elements
        -----------
            Q4
                quadrilateral element with 4 nodes.
        
        3D elements
        -----------
            H8
                hexahedron element with 8 nodes.
        
        attributes
        ----------
        The following attributes are also included in the elementtype definition:
            R
                reduced integration for element, in elementtype[2].
            N
                regular/normal integration for element, in elementtype[2].
            E
                extended integration for element, in elementtype[2].
            PE
                use plane strain for 2D elements, in elementtype[3:5] or [2:4].
            PS
                use plane stress for 2D elements, in elementtype[3:5] or [2:4].
        
        If R, N or E is not given by the user, we assume N. 
        If PE or PS is not given by the user, we assume PE."""
        
        self.elementtype = elementType
        self.elnumber = elNumber
        

    
    def setNodes(self, nodes: list[Node]):
        """Assign the nodes to the element.

        Parameters
        ----------
        nodes
            A list of nodes.
        """

        self._nodes = nodes
        _nodesCoordinates = np.array ( [ n.coordinates for n in nodes  ]   ) # get node coordinates
        self._nodesCoordinates = _nodesCoordinates.transpose() # nodes given column-wise: x-coordinate - y-coordinate

    
    def setProperties(self, elementProperties: np.ndarray):
        """Assign a set of properties to the element.

        Parameters
        ----------
        elementProperties
            A numpy array containing the element properties.

        Possible Parameters
        -------------------
        thickness
            Thickness of 2D elements.
        """
        
        if self.elementtype[0] in ("T","Q"):
            self.th = elementProperties[0] # thickness

    
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

        Possible material parameters
        ----------------------------
        E
            Elasticity module, has to be given first.
        v
            Poisson's ratio, has to be given second.
        """
        self.E = materialProperties[0] # set E
        self.v = materialProperties[1] # set v
    
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
        if self.elementtype[0] in ("Q","T") and (self.elementtype[3:5] in ("PE","") or (len(self.elementtype) == 4 and self.elementtype[2:4] == "PE")): 
            # Ei = elasticity matrix for plane strain for element i (mostly the case for rectangular elements)
            Ei = (self.E*(1-self.v))/((1+self.v)*(1-2*self.v))*np.array([[1, self.v/(1-self.v), 0],
                                                                         [self.v/(1-self.v), 1, 0],
                                                                         [0, 0, (1-2*self.v)/(2*(1-self.v))]])
        elif self.elementtype[0] in ("Q","T") and (self.elementtype[3:5] == "PS" or (len(self.elementtype) == 4 and self.elementtype[2:4] == "PS")):
            # Ei = elasticity matrix for plane stress for element i
            Ei = self.E/(1-self.v**2)*np.array([[1, self.v, 0],
                                                [self.v, 1, 0],
                                                [0, 0, (1-self.v)/2]])
        Ndof = self.nDof
        K = np.reshape(K_,(Ndof,Ndof))
        x = self._nodesCoordinates
        if self.elementtype[0] == "Q": # it's a 2D quadrilateral element
            # [a] matrix that connects strain and displacement derivatives
            a = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 1, 1, 0]])
            # integration points
            if self.elementtype[1:3] in ("4N","8R","4") or (len(self.elementtype) == 4 and self.elementtype[1] == "4"): # assume it's normal integration if it's not given
                Nint = 4
                # get all points
                t = 1/np.sqrt(3)*np.array([-1,-1,1,1]) # get t
                s = 1/np.sqrt(3)*np.array([-1,1,1,-1]) # get s
                # get weights
                w = np.array([1,1,1,1]) # weights to all corresponding points
            elif self.elementtype[1:3] == "4R":
                Nint = 1
                t = np.array([0]) # get t
                s = np.array([0]) # get s
                # get weights
                w = np.array([4]) # weights to all corresponding points
            else:
                raise Exception("Higher order integration not possible yet: WIP")
            if self.elementtype[1] == "4":
                # calc all parameters for the X and Y functions (Q4)
                A = np.array([[1,-1,-1, 1],
                              [1, 1,-1,-1],
                              [1, 1, 1, 1],
                              [1,-1, 1,-1]])
                ax = np.linalg.inv(A) @ np.transpose(x[0])
                ay = np.linalg.inv(A) @ np.transpose(x[1])
                # total Ki for Q4 element
                Ki = np.zeros([8,8])
                for i in range(0,Nint): # for all Gauss points (N in total)
                    # [J] Jacobi matrix (only Q4)
                    J = np.array([[ax[1]+ax[3]*t[i], ay[1]+ay[3]*t[i]],
                                  [ax[2]+ax[3]*s[i], ay[2]+ay[3]*s[i]]])
                    # make inverse of Jacobi
                    invJ = np.linalg.inv(J)
                    # [b] connects displacement derivatives (Q4)
                    b = np.array([[invJ, np.zeros([2,2])],
                                  [np.zeros([2,2]), invJ]])
                    # make [b] what it should actually look like
                    b = b.transpose(0,2,1,3).reshape(4,4)
                    # [h] as a temporary matrix
                    h = np.array([[-1/4*(1-t[i]), 0, 1/4*(1-t[i]), 0, 1/4*(1+t[i]), 0, -1/4*(1+t[i])],
                                  [-1/4*(1-s[i]), 0, -1/4*(1+s[i]), 0, 1/4*(1+s[i]), 0, 1/4*(1-s[i])]])
                    # assemble [c] differentiated shapefunctions (Q4)
                    c = np.vstack([np.hstack([h, np.zeros([2,1])]), np.hstack([np.zeros([2,1]), h])])
                    # [B] for all different s and t
                    Bi = a @ b @ c
                    Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * self.th * w[i]
                    Ki = Ki+Kii
            elif self.elementtype[1] == "8":
                raise Exception("Higher order element not possible yet: WIP")
                # calc all parameters for X and Y (Q8)
        elif self.elementtype[0:2] == "H8": # 3D hexahedron element
            # Ei = elasticity matrix
            Ei = self.E/((1+self.v)*(1-2*self.v))*np.array([[(1-self.v),self.v,self.v,0,0,0],
                                                            [self.v,(1-self.v),self.v,0,0,0],
                                                            [self.v,self.v,(1-self.v),0,0,0],
                                                            [0,0,0,(1-2*self.v)/2,0,0],
                                                            [0,0,0,0,(1-2*self.v)/2,0],
                                                            [0,0,0,0,0,(1-2*self.v)/2]])
            # [a] matrix that connects strain and displacement derivatives
            a = np.array([[1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1],
                          [0,1,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,1,0],
                          [0,0,1,0,0,0,1,0,0]])
            if self.elementtype[1:3] in ("8N","8"):
                Nint = 8
                # get all points
                z = 1/np.sqrt(3)*np.array([1,1,1,1,-1,-1,-1,-1]) # get z (local z)
                t = 1/np.sqrt(3)*np.array([-1,-1,1,1]) # get t
                s = 1/np.sqrt(3)*np.array([-1,1,1,-1]) # get s
                t = np.hstack([t,t])
                s = np.hstack([s,s])
                # get weights
                w = np.array([1,1,1,1,1,1,1,1]) # weights to all corresponding points
            else: # higher order integration follow later
                raise Exception("Higher order integration not possible yet: WIP")
            # calc all parameters for the X, Y and Z functions
            A = np.array([[1,-1,-1,-1,1,1,1,-1],
                          [1,-1,-1,1,1,-1,-1,1],
                          [1,1,-1,1,-1,-1,1,-1],
                          [1,1,-1,-1,-1,1,-1,1],
                          [1,-1,1,-1,-1,-1,1,1],
                          [1,-1,1,1,-1,1,-1,-1],
                          [1,1,1,1,1,1,1,1],
                          [1,1,1,-1,1,-1,-1,-1]])
            ax = np.linalg.inv(A) @ np.transpose(x[0])
            ay = np.linalg.inv(A) @ np.transpose(x[1])
            az = np.linalg.inv(A) @ np.transpose(x[2])
            # total Ki for Hex8 element
            Ki = np.zeros([24,24])
            for i in range(0,Nint): # for all Gauss points (N in total)
                # [J] Jacobi matrix (only Q4)
                J = np.array([[ax[1]+ax[4]*t[i]+ax[6]*z[i]+ax[7]*t[i]*z[i], ay[1]+ay[4]*t[i]+ay[6]*z[i]+ay[7]*t[i]*z[i], az[1]+az[4]*t[i]+az[6]*z[i]+az[7]*t[i]*z[i]],
                              [ax[2]+ax[4]*s[i]+ax[5]*z[i]+ax[7]*s[i]*z[i], ay[2]+ay[4]*s[i]+ay[5]*z[i]+ay[7]*s[i]*z[i], az[2]+az[4]*s[i]+az[5]*z[i]+az[7]*s[i]*z[i]],
                              [ax[3]+ax[5]*t[i]+ax[6]*s[i]+ax[7]*s[i]*t[i], ay[3]+ay[5]*t[i]+ay[6]*s[i]+ay[7]*s[i]*t[i], az[3]+az[5]*t[i]+az[6]*s[i]+az[7]*s[i]*t[i]]])
                # make inverse of Jacobi
                invJ = np.linalg.inv(J)
                # [b] connects displacement derivatives (Hex8)
                b = np.array([[invJ, np.zeros([3,3]), np.zeros([3,3])],
                              [np.zeros([3,3]), invJ, np.zeros([3,3])],
                              [np.zeros([3,3]), np.zeros([3,3]), invJ]])
                # make [b] what it should actually look like
                b = b.transpose(0,2,1,3).reshape(9,9)
                # [h] as a temporary matrix
                h = 1/8*np.array([[-(1-t[i])*(1-z[i]), 0, 0, -(1-t[i])*(1+z[i]), 0, 0, (1-t[i])*(1+z[i]), 0, 0, (1-t[i])*(1-z[i]), 0, 0, -(1+t[i])*(1-z[i]), 0, 0, -(1+t[i])*(1+z[i]), 0, 0, (1+t[i])*(1+z[i]), 0, 0, (1+t[i])*(1-z[i])],
                                  [-(1-s[i])*(1-z[i]), 0, 0, -(1-s[i])*(1+z[i]), 0, 0, -(1+s[i])*(1+z[i]), 0, 0, -(1+s[i])*(1-z[i]), 0, 0, (1-s[i])*(1-z[i]), 0, 0, (1-s[i])*(1+z[i]), 0, 0, (1+s[i])*(1+z[i]), 0, 0, (1+s[i])*(1-z[i])],
                                  [-(1-s[i])*(1-t[i]), 0, 0, (1-s[i])*(1-t[i]), 0, 0, (1+s[i])*(1-t[i]), 0, 0, -(1+s[i])*(1-t[i]), 0, 0, -(1-s[i])*(1+t[i]), 0, 0, (1-s[i])*(1+t[i]), 0, 0, (1+s[i])*(1+t[i]), 0, 0, -(1+s[i])*(1+t[i])]])
                # assemble [c] differentiated shapefunctions (Hex8)
                c = np.vstack([np.hstack([h, np.zeros([3,2])]), np.hstack([np.zeros([3,1]), h, np.zeros([3,1])]), np.hstack([np.zeros([3,2]), h])])
                # [B] for all different s and t
                Bi = a @ b @ c
                Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * w[i]
                Ki = Ki+Kii
        K[:] = Ki # add Ki
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

        pass
