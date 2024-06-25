import numpy as np
from .parameters import *


__all__ = [
    'rl_kecp',
    'rl_kpl',
    'rl_kecl',
    'rl_kve'
]


def mrdivide(B, A):
    '''
    Designed to mimic the backslash notation in MATLAB
    For a MATLAB statement of the form B\A = x or an equation of the form Ax = B
    '''
    shapeA = np.shape(A)
    shapeB = np.shape(B)

    if len(shapeA) == 1:
        A = np.reshape(A, (shapeA[0], 1))
    if len(shapeB) == 1:
        B = np.reshape(B, (1, shapeB[0]))
        
    # lstsq returns a bunch of other information so the [0] index
    #  extracts out only the answer to the equation
    x = np.linalg.lstsq(B, A, rcond = None)[0]
    return x


def rl_kecp(parms:pk_params, acq:acq_params):
    '''
    PK model approximation of P2L3 that is valid when cell uptake of pyruvate is the rate-limiting 
        step in the production of extracellular lactic acid 

    NOTE: This is an experimental addition and is subject to frequent adaptation. This model does not
    apply if kve is the rate-limiting step
    
    Unused PK parameters:
        - kpl
        - kecl
    
    input:
        pk_params object
            -> Can be created by writing parms=pk_params()
            -> Pass non-default parameters as keyword arguments
        acq_params object
            -> Can be created by writing acq=acq_params()
            -> Pass non-default parameters as keyword arguments
        intermediate_pyr as Boolean
            -> True if the rate limiting step is kpl or kecp
            -> False if the rate limiting step is kecl

    output:
        sim_result object with following fields
            -> Mxy: np.array containing transverse magnetization for each metabolite in each compartment
                Mxy[0,:] = intravascular pyruvate
                Mxy[1,:] = intravascular lactate
                Mxy[2,:] = extravascular intermediate metabolite X
                Mxy[3,:] = extravascular lactate
            -> Mz: np.array containing longitudinal magnetization for each metabolite in each compartment
                Mz[0,:] = intravascular pyruvate
                Mz[1,:] = intravascular lactate
                Mz[2,:] = extravascular intermediate metabolite X
                Mz[3,:] = extravascular lactate
            -> pyrSig: np.array containing the total longitudinal magnetization for pyruvate as a function of time
            -> lacSig: np.array containing the total longitudinal magnetization for lactate as a function of time
    
    Note: The only initial conditions used here are pk_params.Le0 and pk_params.Pe0
    '''

    vb = parms.vb
    ve = (1-vb)*parms.vef
    kvedve = parms.kve / ve
    kecpdve = parms.kecp / ve
    ff = np.array([parms.VIFScale*parms.PIF, np.zeros( (acq.ntp,) )])

    # IV magnetization at each segment
    MivSeg = ff
    MxyivSeg = ff *  np.sin( np.radians(acq.FA) )

    # Initialize return variables
    Mxyev = np.zeros( (2,acq.ntp) )
    Mzev = np.zeros( (2,acq.ntp) )
    Mxyiv = np.zeros( (2,acq.ntp) )
    Mziv = np.zeros( (2,acq.ntp) )



    # Calculate non trivial result:
    # [Pev, Lev] = [Pev0, Lev0]*exp(At)+kpl*integral(0,t:exp(A*(t-T))*Piv(T), Liv(T)*dT)
    # A is a matrix with shape (2,2)
    a11 = - ( kvedve + kecpdve + (1/parms.T1pyr))
    a21 = kecpdve
    # a12 = parms.klp  ## <-- need to test if this helps or hurts, where klp is pyruvate efflux from cell
    a12 = 0
    # a22 = - ( 1/parms.T1lac + parms.klp + kvedve)
    a22 = -(1/parms.T1lac + kvedve * parms.vef) # <-- 1/2 used to be kvedve
    # a22 = -( 1/parms.T1lac )  # <-- fit this. What if kvedve is different for the extravasation of pyruvate and revabsorbtion of lactate?
    A = [ [a11, a12], [a21, a22] ]



    dD, P = np.linalg.eig(A)
    # dD should be array of eigenvalues with shape (1,2)
    # P is a matrix containing the eigenvectors of A

    ### Calculate signal evolution in TR ###
    MzevSegIC = np.array( [parms.Pe0, parms.Le0] ) # Set initial conditions
    for i in range(acq.ntp):
        MxyevSeg = np.zeros( (2,) )
        MzevSeg = np.zeros( (2,) )

        TR = acq.TR[i]

        # First account for signal already present and its evolution
        #  Longitudinal magnetization available at the start of each segment
        MzevSeg[:] = MzevSegIC
        # Signal observed at each excitation, at start of segment
        MxyevSeg[:] = np.multiply(
            MzevSegIC, np.sin( np.radians(acq.FA[i]) )
        )

        # At the end of this TR, Mz evolves to contribute to IC for next
        if i < acq.ntp-1: # Don't calculate after last datapoint
            MzevSeg1 = np.exp(dD*TR).flatten() * mrdivide(P, np.multiply(
                    MzevSegIC, np.cos( np.radians(acq.FA[i]) )
                ).T ).flatten()

            # Now calculate new spins flowing into the system
            #  Assume piecewise linear VIF and diagonalize
            dff1 = mrdivide(P, ff[:,i]) # Diag VIF @ start of TR
            dff2 = mrdivide(P, ff[:,i+1]) # Diag VIF @ end of TR            
            
            # Get slope and y-intercept for diagonalized pyruvate forcing function
            b = dff1
            m = (dff2-dff1) / TR

            # At end of this TR inflowing spins will cause
            MevSeg2a = - (b.flatten() / dD) * (1-np.exp(dD*TR))
            MevSeg2b = np.multiply(m.flatten(),
                (-np.divide(TR, dD)- ((1/dD)/dD) ) + (np.exp(dD*TR) * ((1/dD)/dD)) )
            
            # Total signal at end of TR is combination of inflowing and already present signals
            MzevSegIC = np.matmul(P, (MzevSeg1 + kvedve*(MevSeg2a+MevSeg2b)))

        Mxyev[:,i] = np.reshape(MxyevSeg, (2,) )
        Mzev[:,i] = np.reshape(MzevSeg, (2,) )

        Mxyiv[:,i] = MxyivSeg[:,i-1]
        Mziv[:,i] = MivSeg[:,i-1]
    ### END OF CALCULATION LOOP ###
    Mxy = np.array([Mxyiv[0], Mxyiv[1], Mxyev[0], Mxyev[1]])
    Mz = np.array([Mziv[0], Mziv[1], Mzev[0], Mzev[1]])
    pyrSig = vb*Mxy[0] + ve*Mxy[2]
    lacSig = ve*Mxy[3]

    result = sim_result(Mxy, Mz, pyrSig, lacSig)
    return result


def rl_kpl(parms:pk_params, acq:acq_params):
    '''
    PK model approximation of P2L3 that is valid when intracellular chemical reaction of pyruvate is the rate-limiting 
        step in the production of extracellular lactic acid 

    NOTE: This is an experimental addition and is subject to frequent adaptation. This model does not
    apply if kve is the rate-limiting step
    
    Unused PK parameters:
        - kecp
        - kecl
    
    input:
        pk_params object
            -> Can be created by writing parms=pk_params()
            -> Pass non-default parameters as keyword arguments
        acq_params object
            -> Can be created by writing acq=acq_params()
            -> Pass non-default parameters as keyword arguments
        intermediate_pyr as Boolean
            -> True if the rate limiting step is kpl or kecp
            -> False if the rate limiting step is kecl

    output:
        sim_result object with following fields
            -> Mxy: np.array containing transverse magnetization for each metabolite in each compartment
                Mxy[0,:] = intravascular pyruvate
                Mxy[1,:] = intravascular lactate
                Mxy[2,:] = extravascular intermediate metabolite X
                Mxy[3,:] = extravascular lactate
            -> Mz: np.array containing longitudinal magnetization for each metabolite in each compartment
                Mz[0,:] = intravascular pyruvate
                Mz[1,:] = intravascular lactate
                Mz[2,:] = extravascular intermediate metabolite X
                Mz[3,:] = extravascular lactate
            -> pyrSig: np.array containing the total longitudinal magnetization for pyruvate as a function of time
            -> lacSig: np.array containing the total longitudinal magnetization for lactate as a function of time
    
    Note: The only initial conditions used here are pk_params.Le0 and pk_params.Pe0
    '''

    vb = parms.vb
    ve = (1-vb)*parms.vef
    vc = 1-vb-ve
    kvedve = parms.kve / ve
    ff = np.array([parms.VIFScale*parms.PIF, np.zeros( (acq.ntp,) )])

    # IV magnetization at each segment
    MivSeg = ff
    MxyivSeg = ff *  np.sin( np.radians(acq.FA) )

    # Initialize return variables
    Mxyev = np.zeros( (2,acq.ntp) )
    Mzev = np.zeros( (2,acq.ntp) )
    Mxyiv = np.zeros( (2,acq.ntp) )
    Mziv = np.zeros( (2,acq.ntp) )



    # Calculate non trivial result:
    # [Pev, Lev] = [Pev0, Lev0]*exp(At)+kpl*integral(0,t:exp(A*(t-T))*Piv(T), Liv(T)*dT)
    # A is a matrix with shape (2,2)
    a11 = - ( kvedve*(1-parms.vef) + parms.kpl*parms.vef + (1/parms.T1pyr))
    a21 = parms.kpl*parms.vef
    # a12 = parms.klp  ## <-- need to test if this helps or hurts, where klp is backwards rate of reaction
    a12 = 0
    # a22 = - ( 1/parms.T1lac + parms.klp + kvedve)
    a22 = -(1/parms.T1lac + kvedve*parms.vef) # <-- 1/2 used to be kvedve
    # a22 = -( 1/parms.T1lac )  # <-- fit this. What if kvedve is different for the extravasation of pyruvate and revabsorbtion of lactate?
    A = [ [a11, a12], [a21, a22] ]



    dD, P = np.linalg.eig(A)
    # dD should be array of eigenvalues with shape (1,2)
    # P is a matrix containing the eigenvectors of A

    ### Calculate signal evolution in TR ###
    MzevSegIC = np.array( [parms.Pe0, parms.Le0] ) # Set initial conditions
    for i in range(acq.ntp):
        MxyevSeg = np.zeros( (2,) )
        MzevSeg = np.zeros( (2,) )

        TR = acq.TR[i]

        # First account for signal already present and its evolution
        #  Longitudinal magnetization available at the start of each segment
        MzevSeg[:] = MzevSegIC
        # Signal observed at each excitation, at start of segment
        MxyevSeg[:] = np.multiply(
            MzevSegIC, np.sin( np.radians(acq.FA[i]) )
        )

        # At the end of this TR, Mz evolves to contribute to IC for next
        if i < acq.ntp-1: # Don't calculate after last datapoint
            MzevSeg1 = np.exp(dD*TR).flatten() * mrdivide(P, np.multiply(
                    MzevSegIC, np.cos( np.radians(acq.FA[i]) )
                ).T ).flatten()

            # Now calculate new spins flowing into the system
            #  Assume piecewise linear VIF and diagonalize
            dff1 = mrdivide(P, ff[:,i]) # Diag VIF @ start of TR
            dff2 = mrdivide(P, ff[:,i+1]) # Diag VIF @ end of TR            
            
            # Get slope and y-intercept for diagonalized pyruvate forcing function
            b = dff1
            m = (dff2-dff1) / TR

            # At end of this TR inflowing spins will cause
            MevSeg2a = - (b.flatten() / dD) * (1-np.exp(dD*TR))
            MevSeg2b = np.multiply(m.flatten(),
                (-np.divide(TR, dD)- ((1/dD)/dD) ) + (np.exp(dD*TR) * ((1/dD)/dD)) )
            
            # Total signal at end of TR is combination of inflowing and already present signals
            MzevSegIC = np.matmul(P, (MzevSeg1 + kvedve*(MevSeg2a+MevSeg2b)))

        Mxyev[:,i] = np.reshape(MxyevSeg, (2,) )
        Mzev[:,i] = np.reshape(MzevSeg, (2,) )

        Mxyiv[:,i] = MxyivSeg[:,i-1]
        Mziv[:,i] = MivSeg[:,i-1]
    ### END OF CALCULATION LOOP ###
    Mxy = np.array([Mxyiv[0], Mxyiv[1], Mxyev[0], Mxyev[1]])
    Mz = np.array([Mziv[0], Mziv[1], Mzev[0], Mzev[1]])
    pyrSig = vb*Mxy[0] + vc*Mxy[2]
    lacSig = ve*Mxy[3]

    result = sim_result(Mxy, Mz, pyrSig, lacSig)
    return result


def rl_kecl(parms:pk_params, acq:acq_params):
    '''
    PK model approximation of P2L3 that is valid when cell export of lactate is the rate-limiting 
        step in the production of extracellular lactic acid 

    NOTE: This is an experimental addition and is subject to frequent adaptation. This model does not
    apply if kve is the rate-limiting step
    
    Unused PK parameters:
        - kecp
        - kpl
    
    input:
        pk_params object
            -> Can be created by writing parms=pk_params()
            -> Pass non-default parameters as keyword arguments
        acq_params object
            -> Can be created by writing acq=acq_params()
            -> Pass non-default parameters as keyword arguments
        intermediate_pyr as Boolean
            -> True if the rate limiting step is kpl or kecp
            -> False if the rate limiting step is kecl

    output:
        sim_result object with following fields
            -> Mxy: np.array containing transverse magnetization for each metabolite in each compartment
                Mxy[0,:] = intravascular pyruvate
                Mxy[1,:] = intravascular lactate
                Mxy[2,:] = extravascular intermediate metabolite X
                Mxy[3,:] = extravascular lactate
            -> Mz: np.array containing longitudinal magnetization for each metabolite in each compartment
                Mz[0,:] = intravascular pyruvate
                Mz[1,:] = intravascular lactate
                Mz[2,:] = extravascular intermediate metabolite X
                Mz[3,:] = extravascular lactate
            -> pyrSig: np.array containing the total longitudinal magnetization for pyruvate as a function of time
            -> lacSig: np.array containing the total longitudinal magnetization for lactate as a function of time
    
    Note: The only initial conditions used here are pk_params.Le0 and pk_params.Pe0
    '''

    vb = parms.vb
    ve = (1-vb)*parms.vef
    vc = 1-vb-ve
    kvedve = parms.kve / ve
    kecldvc = parms.kecl / vc
    ff = np.array([parms.VIFScale*parms.PIF, np.zeros( (acq.ntp,) )])

    # IV magnetization at each segment
    MivSeg = ff
    MxyivSeg = ff *  np.sin( np.radians(acq.FA) )

    # Initialize return variables
    Mxyev = np.zeros( (2,acq.ntp) )
    Mzev = np.zeros( (2,acq.ntp) )
    Mxyiv = np.zeros( (2,acq.ntp) )
    Mziv = np.zeros( (2,acq.ntp) )



    # Calculate non trivial result:
    # [Pev, Lev] = [Pev0, Lev0]*exp(At)+kpl*integral(0,t:exp(A*(t-T))*Piv(T), Liv(T)*dT)
    # A is a matrix with shape (2,2)
    a11 = - ( kecldvc + (1/parms.T1lac))
    a21 = kecldvc
    # a12 = parms.klp  ## <-- need to test if this helps or hurts, where klp is backwards rate of reaction
    a12 = 0
    # a22 = - ( 1/parms.T1lac + parms.klp + kvedve)
    a22 = -(1/parms.T1lac + kvedve) # <-- 1/2 used to be kvedve
    # a22 = -( 1/parms.T1lac )  # <-- fit this. What if kvedve is different for the extravasation of pyruvate and revabsorbtion of lactate?
    A = [ [a11, a12], [a21, a22] ]



    dD, P = np.linalg.eig(A)
    # dD should be array of eigenvalues with shape (1,2)
    # P is a matrix containing the eigenvectors of A

    ### Calculate signal evolution in TR ###
    MzevSegIC = np.array( [parms.Pe0, parms.Le0] ) # Set initial conditions
    for i in range(acq.ntp):
        MxyevSeg = np.zeros( (2,) )
        MzevSeg = np.zeros( (2,) )

        TR = acq.TR[i]

        # First account for signal already present and its evolution
        #  Longitudinal magnetization available at the start of each segment
        MzevSeg[:] = MzevSegIC
        # Signal observed at each excitation, at start of segment
        MxyevSeg[:] = np.multiply(
            MzevSegIC, np.sin( np.radians(acq.FA[i]) )
        )

        # At the end of this TR, Mz evolves to contribute to IC for next
        if i < acq.ntp-1: # Don't calculate after last datapoint
            MzevSeg1 = np.exp(dD*TR).flatten() * mrdivide(P, np.multiply(
                    MzevSegIC, np.cos( np.radians(acq.FA[i]) )
                ).T ).flatten()

            # Now calculate new spins flowing into the system
            #  Assume piecewise linear VIF and diagonalize
            dff1 = mrdivide(P, ff[:,i]) # Diag VIF @ start of TR
            dff2 = mrdivide(P, ff[:,i+1]) # Diag VIF @ end of TR            
            
            # Get slope and y-intercept for diagonalized pyruvate forcing function
            b = dff1
            m = (dff2-dff1) / TR

            # At end of this TR inflowing spins will cause
            MevSeg2a = - (b.flatten() / dD) * (1-np.exp(dD*TR))
            MevSeg2b = np.multiply(m.flatten(),
                (-np.divide(TR, dD)- ((1/dD)/dD) ) + (np.exp(dD*TR) * ((1/dD)/dD)) )
            
            # Total signal at end of TR is combination of inflowing and already present signals
            MzevSegIC = np.matmul(P, (MzevSeg1 + kvedve*(MevSeg2a+MevSeg2b)))

        Mxyev[:,i] = np.reshape(MxyevSeg, (2,) )
        Mzev[:,i] = np.reshape(MzevSeg, (2,) )

        Mxyiv[:,i] = MxyivSeg[:,i-1]
        Mziv[:,i] = MivSeg[:,i-1]
    ### END OF CALCULATION LOOP ###
    Mxy = np.array([Mxyiv[0], Mxyiv[1], Mxyev[0], Mxyev[1]])
    Mz = np.array([Mziv[0], Mziv[1], Mzev[0], Mzev[1]])
    pyrSig = vb*Mxy[0]
    lacSig = ve*Mxy[3] + vc*Mxy[2]

    result = sim_result(Mxy, Mz, pyrSig, lacSig)
    return result


def rl_kve(parms:pk_params, acq:acq_params):

    '''
    input:
        pk_params object
            -> Can be created by writing parms=pk_params()
            -> Pass non-default parameters as keyword arguments
        acq_params object
            -> Can be created by writing acq=acq_params()
            -> Pass non-default parameters as keyword arguments

    output:
        sim_result object with following fields
            -> Mxy: np array containing transverse magnetization for each metabolite in each compartment
                Mxy[0,:] = pyruvate
                Mxy[1.:] = lactate
            -> Mz: np array containing longitudinal magnetization for each metabolite in each compartment
                Mz[0.:] = pyruvate
                Mz[1,:] = lactate
            -> pyrSig: np array containing the total longitudinal magnetization for pyruvate as a function of time
            -> lacSig: np array containing the total longitudinal magnetization for lactate as a function of time
    
    Note: The only initial condition used here is pk_params.Le0
    '''
        
    # Data from each excitation
    PzSeg = parms.VIFScale*parms.PIF
    PxySeg = PzSeg*np.sin( np.radians(acq.FA) )
    
    # Initialize return variables 
    Mxy = np.zeros( (2, acq.ntp) )
    Mz = np.zeros( (2, acq.ntp) )

    vb = parms.vb
    vef = parms.vef
    ve  = (1-vb)*vef
    kvedve = parms.kve / ve

    # Set up for the following equation
    #  Lac = Lac0 * exp(At) + kpl * inte(0,t: exp(A*(t-T)) *[Pyr(T)] *dT)
    A = -(kvedve*vef + 1/parms.T1lac)

    #### Caluclate signal evolution in each TR ###
    LzSegIC = parms.Le0  # Initial condition before acquisition starts
    for i in range(0, acq.ntp):
        TR = acq.TR[i]
    
        # First account for signal already in the slice and its evolution
        # Longitudinal M available at start of each cycle
        LzSeg = LzSegIC
        # Signal observed at start of each cycle
        LxySeg = LzSegIC * np.sin( np.radians(acq.FA[i]) )
        # LxySeg = np.multiply(LzSegIC, np.sin( np.radians(acq.FA[i]) ))

        # Evolution of this cycle becomes the IC of the next
        if i < acq.ntp-1: # No evolution after last datapoint
            LzSeg1 = np.exp(A*TR)*LzSegIC*np.cos(
                np.radians(acq.FA[i] ))
            # Now account for new spins flowing nto the system during TR
            # Obtain parameters for linear pyruvate forcing function
            b = PzSeg[i]
            m = (PzSeg[i+1]-PzSeg[i]) / TR 

            # Contribution from inflowing spins during this TR
            LzSeg2 = (((m/A + b)* np.exp(A*TR)) - (m*(TR+1/A)) - b)/A
            #Total signal at the end of TR is sum of inflowing and already present
            LzSegIC = LzSeg1 + kvedve*LzSeg2
                
        
            Mxy[:,i] = [PxySeg[i], LxySeg]
            Mz[:,i] = [PzSeg[i], LzSeg]
    ### END OF CALCULATION LOOP ###

    pyrSig = vb*Mxy[0,:]
    lacSig = ve*Mxy[1,:]
    result = sim_result(Mxy, Mz, pyrSig, lacSig)

    return result

