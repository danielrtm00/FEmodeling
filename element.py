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

        if self.elementtype == "Q4": # for 4 nodes
            return 4
        elif self.elementtype =="Q8" or self.elementtype == "HEX8": # for 8 nodes
            return 8

    @property
    def nodes(self) -> int:
        """The list of nodes this element holds"""

        return self._nodes

    @property
    
    def nDof(self) -> int:
        """The total number of degrees of freedom this element has"""
   
        if self.elementtype == "Q4": # for 4 nodes
            return 8
        elif self.elementtype =="Q8" or self.elementtype == "HEX8": # for 8 nodes
            return 16

    @property
    
    def fields(self) -> list[list[str]]:
        """The list of fields per nodes."""
        if self.elementtype == "Q4":
            return [["displacement"],["displacement"],["displacement"],["displacement"]]
        elif self.elementtype == "HEX8":
            return [["displacement"],["displacement"],["displacement"],["displacement"],["displacement"],["displacement"],["displacement"],["displacement"]]

    @property
    
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation pattern for the residual vector and the stiffness matrix to
        aggregate all entries in order to resemble the defined fields nodewise."""

        if self.elementtype == "Q4": # for 4 nodes
            return np.arange(0,8)
        elif self.elementtype =="Q8" or self.elementtype == "HEX8": # for 8 nodes
            return np.arange(0,16)

    @property
    
    def ensightType(self) -> str:
        """The shape of the element in Ensight Gold notation."""
        # ___________________________________???????????????????
        return "quad4"
        

    
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
        """
        
        #"""The following types of elements are currently possible:
                #CSQ1 = constant strain quadrilateral element, 1 integration nodes, plane strain
                #CSQ4 = constant strain quadrilateral element, 4 integration nodes, plane strain"""
        self.elementtype = elementType
        self.elnumber = elNumber
        

    
    def setNodes(self, nodes : list[Node]):
        """Assign the nodes to the element.

        Parameters
        ----------
        nodes
            A list of nodes.
        """

        self._nodes = nodes
        _nodesCoordinates = np.array ( [ n.coordinates for n in nodes  ]   ) 
        self._nodesCoordinates = _nodesCoordinates.transpose() # nodes given column-wise: x-coordinate - y-coordinate

    
    def setProperties(self, elementProperties: np.ndarray):
        """Assign a set of properties to the element.

        Parameters
        ----------
        elementProperties
            A numpy array containing the element properties.
        """
        # thickness
        self.th = elementProperties[0]

    
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
        
        """Evaluate the residual and stiffness for given time, field, and field increment due to a surface load.

        Parameters
        ----------
        P
            The external load vector to be defined.
        K
            The stiffness matrix to be defined.
        U
            The current solution vector.
        dU
            The current solution vector increment.
        time
            Array of step time and total time.
        dTime
            The time increment.
        """
        
        if self.elementtype[0] == "Q": # it's a 2D quadrilateral element
            K = np.reshape(K_, (8,8) )
            Ndof = self.nDof
            x = self._nodesCoordinates
            Nint = int(self.elementtype[1])
            # ________________________________________ -> Ziellösung
            #gapo = gauss_points2D(Nint) # eigentlich damit berechnen - Problem: dauernde Neuberechnung
            # get all points
            #t = gapo["t"] # get t
            #s = gapo["s"] # get s
            # get weights
            #w = gapo["W"] # weights to all corresponding points
            # ________________________________________
            # Ei = elasticity matrix for plane strain for element i (mostly the case for rectangular elements)
            Ei = (self.E*(1-self.v))/((1+self.v)*(1-2*self.v))*np.array([[1, self.v/(1-self.v), 0],
                                                                         [self.v/(1-self.v), 1, 0],
                                                                         [0, 0, (1-2*self.v)/(2*(1-self.v))]])
            # [a] matrix that connects strain and displacement derivatives
            a = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 1, 1, 0]])
            # ________________________________________ -> Übergangslösung
            if Nint == 4:
                # get all points
                t = 1/np.sqrt(3)*np.array([-1,-1,1,1]) # get t
                s = 1/np.sqrt(3)*np.array([-1,1,1,-1]) # get s
                # get weights
                w = np.array([1,1,1,1]) # weights to all corresponding points
            elif Nint == 1:
                t = np.array([0]) # get t
                s = np.array([0]) # get s
                # get weights
                w = np.array([4]) # weights to all corresponding points
            if self.elementtype == "Q4":
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
            elif self.elementtype == "Q8":
                print("not possible yet: WIP")
                raise SystemExit()
                # calc all parameters for X and Y (Q8)
        elif self.elementtype == "HEX8": # 3D hexahedron element
            K = np.reshape(K_,(24,24))
            Ndof = self.nDof
            x = self._nodesCoordinates
            Nint = int(self.elementtype[3])
            # Ei = elasticity matrix for plane strain for element i (mostly the case for rectangular elements)
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
            if Nint == 8:
                # get all points
                z = 1/np.sqrt(3)*np.array([1,1,1,1,-1,-1,-1,-1]) # get z (local z)
                t = 1/np.sqrt(3)*np.array([-1,-1,1,1]) # get t
                s = 1/np.sqrt(3)*np.array([-1,1,1,-1]) # get s
                t = np.hstack([t,t])
                s = np.hstack([s,s])
                # get weights
                w = np.array([1,1,1,1,1,1,1,1]) # weights to all corresponding points
            else: # higher order elements follow later
                print("not possible yet: WIP")
                raise SystemExit()
            # calc all parameters for the X, Y and Z functions
            A = np.array([[1,-1,-1,1,1,-1,-1,1],
                          [1,-1,-1,-1,1,1,1,-1],
                          [1,-1,1,-1,-1,-1,1,1],
                          [1,-1,1,1,-1,1,-1,-1],
                          [1,1,-1,1,-1,-1,1,-1],
                          [1,1,-1,-1,-1,1,-1,1],
                          [1,1,1,-1,1,-1,-1,-1],
                          [1,1,1,1,1,1,1,1]])
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
                b = np.array([[invJ, np.zeros([3,3])],
                              [np.zeros([3,3]), invJ]])
                # make [b] what it should actually look like
                b = b.transpose(0,2,1,3).reshape(6,6)
                # [h] as a temporary matrix
                h = 1/8*np.array([[-(1-t[i])*(1+z[i]), 0, 0, -(1-t[i])*(1-z[i]), 0, 0, -(1+t[i])*(1-z[i]), 0, 0, -(1+t[i])*(1+z[i]), 0, 0, (1-t[i])*(1+z[i]), 0, 0, (1-t[i])*(1-z[i]), 0, 0, (1+t[i])*(1-z[i]), 0, 0, (1+t[i])*(1+z[i])],
                                  [-(1-s[i])*(1+z[i]), 0, 0, -(1-s[i])*(1-z[i]), 0, 0, (1-s[i])*(1-z[i]), 0, 0, (1-s[i])*(1+z[i]), 0, 0, -(1+s[i])*(1+z[i]), 0, 0, -(1+s[i])*(1-z[i]), 0, 0, (1+s[i])*(1-z[i]), 0, 0, (1+s[i])*(1+z[i])],
                                  [(1-s[i])*(1-t[i]), 0, 0, -(1-s[i])*(1-t[i]), 0, 0, -(1-s[i])*(1+t[i]), 0, 0, (1-s[i])*(1+t[i]), 0, 0, (1+s[i])*(1-t[i]), 0, 0, -(1+s[i])*(1-t[i]), 0, 0, -(1+s[i])*(1+t[i]), 0, 0, (1+s[i])*(1+t[i])]])
                # assemble [c] differentiated shapefunctions (Hex8)
                c = np.vstack([np.hstack([h, np.zeros([3,2])]), np.hstack([np.zeros([3,1]), h, np.zeros([3,1])]), np.hstack([np.zeros([3,2]), h])])
                # [B] for all different s and t
                Bi = a @ b @ c
                Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * self.th * w[i]
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
