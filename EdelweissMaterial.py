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

    def computeStress(
        self,
        stress: np.ndarray,
        dStressdStrain: np.ndarray,
        dStrain: np.ndarray,
        time: float,
        dTime: float,
        dim: int,
        plStrain: bool,
    ):
        """Computes the stresses in the element.

        Parameters
        ----------
        stress
            Vector containing the stresses in the element.
        dStressdStrain
            Matrix containing dStress/dStrain.
        dStrain
            Strain vector increment at time step t to t+dTime.
        time
            Array of step time and total time.
        dTime
            Current time step size.
        dim
            Dimension of the model (2D or 3D).
        plStrain
            Checks if the elasticity matrix should be plane strain or plane stress.
        """

        try:
            self.strain
        except AttributeError:
            if dim == 2:
                self.strain = np.zeros([3])
                self.strainOld = np.zeros([3])
            elif dim == 3:
                self.strain = np.zeros([6])
                self.strainOld = np.zeros([6])
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
            # calculate new strain
            if time[-1] == 0:  # first step
                self.strain = dStrain  # reset to first value
                self.strainOld = dStrain  # = also old value
            else:
                try:
                    self._time
                except AttributeError:
                    self._time = time[-1] # save old time
                if self._time == time[-1]:  # total time stayed the same, step failed
                    self.strain = self.strainOld + dStrain  # reset to old value and add step
                elif self._time != time[-1]:  # total time changed, step successful
                    self.strainOld = self.strain # set old value to successful step
                    self.strain += dStrain  # add to get new value
                    self._time = time[-1] # total time now
            #pdb.set_trace()
            # calculate new stress with new strain
            self.stress = Ei @ self.strain[..., None]
            # for linear elastic it's just the elasticity matrix
            self.dStressdStrain = Ei
        else:
            raise Exception("This material type doesn't exist (yet). Selected material: " + self.materialname)
        # if one needs the direct output:
        # return {"stress": stress, "strain": dStrain, "dStressdStrain": dStressdStrain}
