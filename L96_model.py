""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved 
"""

import numpy as np 

def L96_eq1_xdot(X, t, F):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96, equation 1:
        d/dt X[k] = -X[k-2] X[k-1] + X[k-1] X[k+1] - X[k] + F

    Args:
        X : Values of X variables at the current time step
        t : Time
        F : Forcing term
    Returns:
        dXdt : Array of X time tendencies
    """

    K = len(X)
    Xdot = np.zeros(K)
    
    for k in range(K):
        Xdot[k] = ( X[(k+1)%K] - X[k-2] ) * X[k-1] - X[k] + F
    return Xdot

def L96_eq2_xdot(X, t, Y, F, h, b, c):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96, equation 2:
        d/dt X[k] = -X[k-1] ( X[k-2] - X[k+1] ) - X[k] + F - h.c/b sum_j Y[j,k]

    Args:
        X : Values of X variables at the current time step
        t : Time
        Y : Values of Y variables at the current time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio
    Returns:
        dXdt : Array of X time tendencies
    """

    JK,K = len(Y),len(X)
    J = JK//K
    assert JK==J*K, "X and Y have incompatible shapes"
    Xdot = np.zeros(K)
    hcb = (h*c)/b

    Ysummed = Y.reshape((K,J)).sum(axis=-1)
    
    for k in range(K):
        Xdot[k] = ( X[(k+1)%K] - X[k-2] ) * X[k-1] - X[k] + F - hcb * Ysummed[k]
    return Xdot

def L96_eq3_ydot(Y, t, X, h, b, c):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96, equation 2:
      d/dt X[k] = -X[k-1] ( X[k-2] - X[k+1] ) - X[k] + F - h.c/b sum_j Y[j,k]

    Args:
        Y : Values of Y variables at the current time step
        t : Time
        X : Values of X variables at the current time step
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio
    Returns:
        dXdt : Array of X time tendencies
    """

    JK,K = len(Y),len(X)
    J = JK//K
    assert JK==J*K, "X and Y have incompatible shapes"
    Ydot = np.zeros((JK))
    hcb = (h*c)/b
 
    for j in range(JK):
            k = j//J
            Ydot[j] = -c * b * Y[(j+1)%JK] * ( Y[(j+2)%JK] - Y[j-1] ) - c * Y[j] + hcb * X[k]
    return Ydot

# Time-stepping methods ##########################################################################################

def EulerFwd(fn, dt, X, t, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the Euler forward method.

    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        t  : Time at beginning of time step
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """
    return X + dt * fn(X, t, *params)

def RK2(fn, dt, X, t, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the second order Runge-Kutta method.

    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        t  : Time at beginning of time step
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """
    X1 = X + 0.5 * dt * fn(X, t, *params)
    return X + dt * fn(X1, t+0.5*dt, *params)

def RK4(fn, dt, X, t, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,...) using the fourth order Runge-Kutta method.

    Args:
        fn     : The function returning the time rate of change of model variables X
        dt     : The time step
        X      : Values of X variables at the current time, t
        t      : Time at beginning of time step
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """
    Xdot1 = fn(X, t, *params)
    Xdot2 = fn(X+0.5*dt*Xdot1, t+0.5*dt, *params)
    Xdot3 = fn(X+0.5*dt*Xdot2, t+0.5*dt, *params)
    Xdot4 = fn(X+dt*Xdot3, t+dt, *params)
    return X + (dt/6.) * ( ( Xdot1 + Xdot4 ) + 2. * ( Xdot2 + Xdot3 ) )

def EulerFwd_XY(Xfn, Yfn, dt, X, Y, t, F, h, b, c):
    """
    Calculate the new state X(n+1), Y(n+1) for d/dt X = fn(X,t,Y...) and d/dt Y = Xfn(Y,t,X,...) using the Euler forward method.

    Args:
        Xfn     : The function returning the time rate of change of model variables X
        Yfn     : The function returning the time rate of change of model variables Y
        dt      : The time step
        X       : Values of X variables at the current time, t
        Y       : Values of Y variables at the current time, t
        t       : Time at beginning of time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio

    Returns:
        X at t+dt
    """
    return X + dt * Xfn(X, t, Y, F, h, b, c), Y + dt * Yfn(Y, t, X, h, b, c)

def RK2_XY(Xfn, Yfn, dt, X, Y, t, F, h, b, c):
    """
    Calculate the new state X(n+1), Y(n+1) for d/dt X = fn(X,t,Y...) and d/dt Y = Xfn(Y,t,X,...) using the second order Runge-Kutta method.

    Args:
        Xfn     : The function returning the time rate of change of model variables X
        Yfn     : The function returning the time rate of change of model variables Y
        dt      : The time step
        X       : Values of X variables at the current time, t
        Y       : Values of Y variables at the current time, t
        t       : Time at beginning of time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio

    Returns:
        X,Y at t+dt
    """
    X1 = X + 0.5 * dt * Xfn(X, t, Y, F, h, b, c)
    Y1 = Y + 0.5 * dt * Yfn(Y, t, X, h, b, c)
    return X + dt * Xfn(X1, t+0.5*dt, Y1, F, h, b, c), Y + dt * Yfn(Y1, t+0.5*dt, X1, h, b, c)

def RK4_XY(Xfn, Yfn, dt, X, Y, t, F, h, b, c):
    """
    Calculate the new state X(n+1), Y(n+1) for d/dt X = fn(X,t,Y...) and d/dt Y = Xfn(Y,t,X,...) using the fourth order Runge-Kutta method.

    Args:
        Xfn     : The function returning the time rate of change of model variables X
        Yfn     : The function returning the time rate of change of model variables Y
        dt      : The time step
        X       : Values of X variables at the current time, t
        Y       : Values of Y variables at the current time, t
        t       : Time at beginning of time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio

    Returns:
        X,Y at t+dt
    """
    Xdot1 = Xfn(X, t, Y, F, h, b, c)
    Ydot1 = Yfn(Y, t, X, h, b, c)
    Xdot2 = Xfn(X+0.5*dt*Xdot1, t+0.5*dt, Y+0.5*dt*Ydot1, F, h, b, c)
    Ydot2 = Yfn(Y+0.5*dt*Ydot1, t+0.5*dt, X+0.5*dt*Xdot1, h, b, c)
    Xdot3 = Xfn(X+0.5*dt*Xdot2, t+0.5*dt, Y+0.5*dt*Ydot2, F, h, b, c)
    Ydot3 = Yfn(Y+0.5*dt*Ydot2, t+0.5*dt, X+0.5*dt*Xdot2, h, b, c)
    Xdot4 = Xfn(X+dt*Xdot3, t+dt, Y+dt*Ydot3, F, h, b, c)
    Ydot4 = Yfn(Y+dt*Ydot3, t+dt, X+dt*Xdot3, h, b, c)
    return X + (dt/6.) * ( ( Xdot1 + Xdot4 ) + 2. * ( Xdot2 + Xdot3 ) ), Y + (dt/6.) * ( ( Ydot1 + Ydot4 ) + 2. * ( Ydot2 + Ydot3 ) )

# Model integrators #############################################################################################

def integrator_1d(fn, method, dt, X0, nt, *params):
    """
    Integrates forward-in-time the model "fn" using the integration "method". Returns the full history with
    nt+1 values including initial conditions for n=0. The model "fn" is required to have one vector of state
    variables, X, and take the form fn(X, t, *params) where t is current model time.
    
    Args:
        fn     : The function returning the time rate of change of model variables X
        method : The time-stepping method that returns X(n+1) givein X(n)
        dt     : The time step
        X0     : Values of X variables at the current time
        nt     : Number of forwards steps
        params : All other arguments that should be passed to fn

    Returns:
        X[:,:], time[:] : the full history X[n,k] at times t[n]
    """    
    time, hist = np.zeros((nt+1)), np.zeros((nt+1,len(X0)))
    X = X0.copy()
    hist[0,:] = X
    for n in range(nt):
        X = method( fn, dt, X, n*dt, *params )
        hist[n+1], time[n+1] = X, dt*(n+1)
    return hist, time

def integrator_2d(Xfn, Yfn, method, dt, X0, Y0, nt, *params):
    """
    Integrates forward-in-time the model "fn" using the integration "method". Returns the full history with
    nt+1 values including initial conditions for n=0. The model "fn" is required to have one vector of state
    variables, X, and take the form fn(X, t, *params) where t is current model time.
    
    Args:
        Xfn     : The function returning the time rate of change of model variables X
        Yfn     : The function returning the time rate of change of model variables Y
        method : The time-stepping method that returns X(n+1) givein X(n)
        dt     : The time step
        X0     : Values of X variables at the current time
        Y0     : Values of Y variables at the current time
        nt     : Number of forwards steps
        params : All other arguments that should be passed to Xfn, Yfn

    Returns:
        X[:,:], time[:] : the full history X[n,k] at times t[n]
    """    
    time, xhist, yhist = np.zeros((nt+1)), np.zeros((nt+1,len(X0))), np.zeros((nt+1,len(Y0)))
    X,Y = X0.copy(), Y0.copy()
    xhist[0,:] = X
    yhist[0,:] = Y
    for n in range(nt):
        X,Y = method( Xfn, Yfn, dt, X, Y, n*dt, *params )
        xhist[n+1], yhist[n+1], time[n+1] = X, Y, dt*(n+1)
    return xhist, yhist, time
