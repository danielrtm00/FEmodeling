import numpy as np
import numpy.linalg as lin
import pdb


class EdelweissMaterial:
    def __init__(self, materialName: str, materialProperties: np.ndarray):
        """Define the material properties.

        Parameters
        ----------
        materialName
            A string identifying the requested material.
        materialProperties
            The numpy array containing the material properties.

        Attributes
        ----------
        linearelastic
            Use linear elastic material. (= materialName)"""

        self.materialname = materialName
        if self.materialname.lower() == "linearelastic":
            self.E = materialProperties[0]  # set E
            self.v = materialProperties[1]  # set v
        else:
            raise Exception("This material type doesn't exist (yet). Current material: " + self.materialname)

    def computeElasticity(self, dim: int, plStrain: bool):
        """Compute the elasticity matrix for the material depending on the element. 
        This method is only used for the calculation of the stiffness matrix.

        Parameters
        ----------
        elementType
            A string identifying the requested element formulation as shown in element."""

        if dim == 2 and plStrain:
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
        elif dim == 2 and not plStrain:
            # Ei = elasticity matrix for plane stress for element i
            Ei = self.E / (1 - self.v**2) * np.array([[1, self.v, 0], [self.v, 1, 0], [0, 0, (1 - self.v) / 2]])
        elif dim == 3:
            # 3D elasticity matrix for Hexa8
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
        return Ei

    def computeStress(
        self,
        stress: np.ndarray,
        dStressdStrain: np.ndarray,
        strain: np.ndarray,
        time: float,
        dTime: float,
        dU: np.ndarray,
        U: np.ndarray,
        dim: int,
        plStrain: bool,
    ):
        """Computes the stresses in the element.

        Parameters
        ----------
        stress
            Matrix containing the stresses in the element.
        dStressdStrain
            Matrix containing dStress/dStrain.
        strain
            Strain matrix at time t = n before and t = n+1 after this function.
        time
            New time after this step.
        dTime
            Current time step size.
        dU
            Change in displacement in this time step.
        """

        if self.materialname.lower() == "linearelastic":
            if dim == 2 and plStrain:
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
            elif dim == 2 and not plStrain:
                # Ei = elasticity matrix for plane stress for element i
                Ei = self.E / (1 - self.v**2) * np.array([[1, self.v, 0], [self.v, 1, 0], [0, 0, (1 - self.v) / 2]])
            elif dim == 3:
                # 3D elasticity matrix for Hexa8
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
            # calculate new stress with new strain
            stress = Ei @ strain
            # for linear elastic it's just the elasticity matrix
            dStressdStrain = Ei
        else:
            raise Exception("This material type doesn't exist (yet). Current material: " + self.materialname)
        # if one needs the direct output:
        return {"stress": stress, "strain": strain, "dStressdStrain": dStressdStrain}
