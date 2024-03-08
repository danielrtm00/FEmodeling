# import packages
import numpy as np

# make function that calculates the Gauss points
def gauss_points1D(N):
    # calculate 1D order of Gauss points with Newton-Raphson method
    # N = number of points on line
    # for N = 1 it's different:
    if N == 1:
        return np.array([[0.],[2.]])
    N = N-1
    # create equal distribution of points
    y = np.linspace(-1,1,N+1)
    P = np.linspace(0,N,N+1)
    # guess the ys
    y = np.cos((2*P+1)*np.pi/(2*N+2))+(0.27/(N+1))*np.sin(np.pi*y*N/(N+2))
    # make empty matrices
    L = np.zeros([N+2,N+1])
    # Compute the zeros of the N+1 Legendre Polynomial
    yold = 2
    # Iterate until new points are uniformly within epsilon of old points
    while np.max(np.abs(y-yold))>np.finfo(np.float64).eps:
        L[0][:]=1
        L[1][:]=np.transpose(y)
        for k in range(2,N+2):
            L[k][:]=((2*k-1)*y*L[k-1][:]-(k-1)*L[k-2][:])/k
        Lp = (N+2)*(L[N][:]-y*L[N+1][:])/(1-y**2)
        yold=y
        y=yold-L[N+1][:]/Lp
    # Compute the weights
    w=2/((1-y**2)*Lp**2)*((N+2)/(N+1))**2
    # return with first row as coordinates and second row with weights
    return np.vstack([y,w])

def gauss_points2D(N):
    # calculate the Gauss points and coordinates for rectangular elements in 2D
    # N = number of points on 2D element (rectangle)
    # calculate number of elements on 1D
    if (np.sqrt(N) % 1) == 0: # Q1, Q4, Q9, Q16 and so on
        # transfer number of elements on 2D to 1D
        N1D = int(np.sqrt(N))
        X = gauss_points1D(N1D) # in these cases 1D is correct
        t = np.zeros([N,1]) # initialize empty matrix for t
        s = np.zeros([N,1]) # initialize empty matrix for s
        W = np.zeros([N,1]) # initialize empty matrix for weights
        k = 0
        # make 2D rectangle points
        for i in range(0,N1D):
            for j in range(0,N1D):
                # round to 29th decimal (slightly more than computer precision)
                t[k] = np.round(-X[0][i], 29)
                s[k] = np.round(-X[0][j], 29)
                W[k] = X[1][i]*X[1][j]
                k += 1
    else: # non conformity quadrilateral elements not computable with this code
        print("Unfortunately the calculation is only possible if sqrt(Nint) = int.")
        raise SystemExit()
    # check if sum of weights = 4
    if np.round(np.sum(W))==4:
        print("Calculation of Gauss points and weights was successful.")
    else:
        print("Please check for errors in weights.")
    return {"t": t.flatten(),"s": s.flatten(), "W": W.flatten()}

def gauss_points3D(N):
    # calculate the Gauss points and coordinates for rectangular elements in 3D
    # N = number of points on 3D element (rectangle)
    # calculate number of elements on 1D
    if (np.cbrt(N) % 1) == 0: # Q1, Q4, Q9, Q16 and so on
        # transfer number of elements on 2D to 1D
        N1D = int(np.cbrt(N))
        X = gauss_points1D(N1D) # in these cases 1D is correct
        t = np.zeros([N,1]) # initialize empty matrix for t
        s = np.zeros([N,1]) # initialize empty matrix for s
        z = np.zeros([N,1]) # initialize empty matrix for z
        W = np.zeros([N,1]) # initialize empty matrix for weights
        k = 0
        # make 3D rectangle points
        for i in range(0,N1D):
            for j in range(0,N1D):
                for u in range(0,N1D):
                    # round to 29th decimal (slightly more than computer precision)
                    t[k] = np.round(-X[0][i], 29)
                    s[k] = np.round(-X[0][j], 29)
                    z[k] = np.round(-X[0][u], 29)
                    W[k] = X[1][i]*X[1][j]*X[1][u]
                    k += 1
    else: # non conformity quadrilateral elements not computable with this code
        print("Unfortunately the calculation is only possible if cbrt(Nint) = int.")
        raise SystemExit()
    # check if sum of weights = 8
    if np.round(np.sum(W))==8:
        print("Calculation of Gauss points and weights was successful.")
    else:
        print("Please check for errors in weights: W="+str(np.round(np.sum(W)))+".")
    return {"z": z.flatten(), "t": t.flatten(),"s": s.flatten(), "W": W.flatten()}