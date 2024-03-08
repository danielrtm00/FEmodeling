import numpy as np

class ElementConstruction():
    def __init__(self, x, Nnodes, dim):
        '''Initalize the data.
        
        Parameters
        ----------
        x
            A matrix containing the points of an isometric element [-1, 1].
        Nnodes
            Number of nodes on the element.
        dim
            Dimension of the element (2 or 3D).'''
        
        self._x = x # isoparametric coordinates
        self._Nnodes = Nnodes
        self._dim = dim

    def calc_A(self):
        '''Calculates the A matrix for a certain element.'''

        A = np.zeros([self._Nnodes,self._Nnodes])
        if self._Nnodes == 20 and self._dim == 3:
            for i in range(20): # for point i
                s = x[i][0]
                t = x[i][1]
                z = x[i][2]
                A[i] = np.array([1, s, t, z, s*t, t*z, s*z, s**2, t**2, z**2, s**2*t, s*t**2, t**2*z, z**2*t, z**2*s, z*s**2, s*t*z, s**2*t*z, s*t**2*z, s*t*z**2])
        elif self._Nnodes == 8 and self._dim == 3:
            for i in range(8): # for point i
                s = x[i][0]
                t = x[i][1]
                z = x[i][2]
                A[i] = np.array([1, s, t, z, s*t, t*z, s*z, s*t*z])
        elif self._Nnodes == 4 and self._dim == 2:
            for i in range(4):
                s = x[i][0]
                t = x[i][1]
                A[i] = np.array([1, s, t, s*t])
        elif self._Nnodes == 8 and self._dim == 2:
            for i in range(8):
                s = x[i][0]
                t = x[i][1]
                A[i] = np.array([1, s, t, s*t, s**2, t**2, s**2*t, t**2*s])
        else:
            raise Exception("Element dimension "+str(self._dim)+" and/or number of nodes "+str(self._Nnodes)+"invalid.")
        np.savetxt('matrixA.txt', A, fmt='%2i', delimiter=',', newline='],\n[')
        print("Matrix A has been saved in 'matrixA.txt'.")
        return A
    
    def calc_h(self):
        '''Calculates the h matrix for a certain element.'''

        

# for testing (Hexa20)
x = np.array([[-1, -1, -1],
              [-1,-1,1],
              [1,-1,1],
              [1,-1,-1],
              [-1,1,-1],
              [-1,1,1],
              [1,1,1],
              [1,1,-1],
              [-1,-1,0],
              [0,-1,1],
              [1,-1,0],
              [0,-1,-1],
              [-1,1,0],
              [0,1,1],
              [1,1,0],
              [0,1,-1],
              [-1,0,-1],
              [-1,0,1],
              [1,0,1],
              [1,0,-1]])

x = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
nodes = 4
dim = 2

constr = ElementConstruction(x,nodes,dim)
print(constr.calc_A())
print(constr.calc_h())