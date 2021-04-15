import numpy as np
from numba import jit

@jit
def Lin3dvar(ub,w,H,R,B,opt):
    
    # The solution of the 3DVAR problem in the linear case requires 
    # the solution of a linear system of equations.
    # Here we utilize the built-in numpy function to do this.
    # Other schemes can be used, instead.
    
    if opt == 1: #model-space approach
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T)@Ri@H
        b = Bi@ub + (H.T)@Ri@w
        ua = np.linalg.solve(A,b) #solve a linear system 
    
    elif opt == 2: #model-space incremental approach
        
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T)@Ri@H
        b = (H.T)@Ri@(w-H@ub)
        ua = ub + np.linalg.solve(A,b) #solve a linear system 
        
        
    elif opt == 3: #observation-space incremental approach
    
        A = R + H@B@(H.T)
        b = (w-H@ub)
        ua = ub + B@(H.T)@np.linalg.solve(A,b) #solve a linear system
        
    return ua

@jit
def EnKF(ubi,w,H,R,B):
    
    # The analysis step for the (stochastic) ensemble Kalman filter 
    # with virtual observations

    n,N = ubi.shape # n is the state dimension and N is the size of ensemble
    m = w.shape[0] # m is the size of measurement vector
    mR = R.shape[0]
    nB = B.shape[0]
    mH, nH = H.shape
    
    assert m==mR, "obseravtion and obs_cov_matrix have incompatible size"
    assert nB==n, "state and state_cov_matrix have incompatible size"
    assert m==mH, "obseravtion and obs operator have incompatible size"
    assert n==nH, "state and obs operator have incompatible size"

    # compute the mean of forecast ensemble
#     ub = ubi.sum(axis=-1)/N   
    # compute Jacobian of observation operator at ub
    # compute Kalman gain
    D = H@B@H.T + R
    K = B @ H.T @ np.linalg.inv(D)
            
#     wi = np.zeros((m,N))
#     uai = np.zeros((n,N))
    wi=w.repeat(N).reshape(m,N)+np.random.standard_normal((m,N))
#     for i in range(N):
        # create virtual observations
#         wi[:,i] = w + np.random.multivariate_normal(np.zeros(m), R)
        # compute analysis ensemble
    uai = ubi + K @ (wi-H@ubi)
    # compute the mean of analysis ensemble
#     ua = ubi.sum(axis=-1)/N    
    # compute analysis error covariance matrix
#     P = (1/(N-1)) * (uai - ua.reshape(-1,1)) @ (uai - ua.reshape(-1,1)).T
#     P = np.cov(uai)
    return uai