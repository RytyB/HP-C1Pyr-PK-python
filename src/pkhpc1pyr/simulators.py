import numpy as np
from .parameters import *


__all__ = [
    'P2L1',
    'P2L2',
    'P2L3',
    'simp_P2L3'
    ]


def mldivide(B, A):
    '''
    Designed to mimic the forward slash notation in MATLAB
    For a MATLAB statement of the form B/A = x or an equation of the form xA = B
    '''
    shapeA = np.shape(A)
    shapeB = np.shape(B)

    if len(shapeA) == 1:
        A = np.reshape(A, (1, shapeA[0]))
    if len(shapeB) == 1:
        B = np.reshape(B, (shapeB[0], 1))

    # lstsq returns a bunch of other information so the [0] index 
    #  selects out only the answer to the equation
    almostX = np.linalg.lstsq(A.T, B.T, rcond = None)[0]
    x = almostX.T
    return x

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

def P2L1(parms, acq):

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

    # Set up for the following equation
    #  Lac = Lac0 * exp(At) + kpl * inte(0,t: exp(A*(t-T)) *[Pyr(T)] *dT)
    A = -(parms.klp + 1/parms.T1lac)

    #### Caluclate signal evolution in each TR ###
    LzSegIC = parms.Le0  # Initial condition before acquisition starts
    for i in range(0, acq.ntp):
        TR = acq.TR[i]
    
        # First account for signal already in the slice and its evolution
        # Longitudinal M available at start of each cycle
        LzSeg = LzSegIC
        # Signal observed at start of each cycle
        LxySeg = np.multiply(LzSegIC, np.sin( np.radians(acq.FA[i]) ))

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
            LzSegIC = LzSeg1 + parms.kpl * LzSeg2
                
        
        Mxy[:,i] = [PxySeg[i], LxySeg]
        Mz[:,i] = [PzSeg[i], LzSeg]
    ### END OF CALCULATION LOOP ###

    pyrSig = Mz[0,:]
    lacSig = Mz[1,:]
    result = sim_result(Mxy, Mz, pyrSig, lacSig)

    return result

def P2L2(parms, acq):
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
            -> Mxy: np.array containing transverse magnetization for each metabolite in each compartment
                Mxy[0.:] = intravascular pyruvate
                Mxy[1,:] = intravascular lactate
                Mxy[2.:] = extravascular pyruvate
                Mxy[3,:] = extravascular lactate
            -> Mz: np.array containing longitudinal magnetization for each metabolite in each compartment
                Mz[0.:] = intravascular pyruvate
                Mz[1,:] = intravascular lactate
                Mz[2.:] = extravascular pyruvate
                Mz[3,:] = extravascular lactate
            -> pyrSig: np.array containing the total longitudinal magnetization for pyruvate as a function of time
            -> lacSig: np.array containing the total longitudinal magnetization for lactate as a function of time
    
    Note: The only initial conditions used here are pk_params.Le0 and pk_params.Pe0
          The only volume fraction used is pk_params.vb
    '''

    kvedve = parms.kve / (1-parms.vb)
    ff = np.array([parms.VIFScale*parms.PIF, np.zeros( (acq.ntp,) )])

    # IV magnetization at each segment
    MivSeg = ff
    MxyivSeg = ff *  np.sin( np.radians(acq.FA) )

    # Initialize return variables
    Mxyev = np.zeros( (2,acq.ntp) )
    Mzev = np.zeros( (2,acq.ntp) )
    vb = parms.vb
    Mxyiv = np.zeros( (2,acq.ntp) )
    Mziv = np.zeros( (2,acq.ntp) )

    # Calculate non trivial result:
    # [Pev, Lev] = [Pev0, Lev0]*exp(At)+kpl*integral(0,t:exp(A*(t-T))*Piv(T), Liv(T)*dT)
    # A is a matrix with shape (2,2)
    a11 = - ( kvedve + parms.kpl + (1/parms.T1pyr) + parms.extraPyrLoss)
    a12 = parms.klp
    a21 = parms.kpl
    a22 = - ( kvedve + parms.klp + (1/parms.T1lac) + parms.extraLacLoss)
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

        Mxyev[:,i-1] = np.reshape(MxyevSeg, (2,) )
        Mzev[:,i-1] = np.reshape(MzevSeg, (2,) )

        Mxyiv[:,i-1] = MxyivSeg[:,i-1]
        Mziv[:,i-1] = MivSeg[:,i-1]
    ### END OF CALCULATION LOOP ###
    Mxy = np.array([Mxyiv[0], Mxyiv[1], Mxyev[0], Mxyiv[1]])
    Mz = np.array([Mziv[0], Mziv[1], Mzev[0], Mzev[1]])
    pyrSig = vb*Mz[0] + (1-vb)*Mz[2]
    lacSig = vb*Mz[1] + (1-vb)*Mz[3]

    result = sim_result(Mxy, Mz, pyrSig, lacSig)

    return result

def P2L3(parms, acq):
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
            -> Mxy: np.array containing transverse magnetization for each metabolite in each compartment
                Mxy[0.:] = intravascular pyruvate
                Mxy[1,:] = intravascular lactate
                Mxy[2.:] = extravascular pyruvate
                Mxy[3,:] = extravascular lactate
            -> Mz: np.array containing longitudinal magnetization for each metabolite in each compartment
                Mz[0.:] = intravascular pyruvate
                Mz[1,:] = intravascular lactate
                Mz[2.:] = extravascular pyruvate
                Mz[3,:] = extravascular lactate
            -> pyrSig: np.array containing the total longitudinal magnetization for pyruvate as a function of time
            -> lacSig: np.array containing the total longitudinal magnetization for lactate as a function of time
    
    Note: The only initial conditions used here are pk_params.Le0 and pk_params.Pe0
          The only volume fraction used is pk_params.vb
    '''

    vb = parms.vb
    ve = (1-vb)*parms.vef
    vc = 1-vb-ve
    R1Pyr = 1 / parms.T1pyr
    R1Lac = 1 / parms.T1lac
    kvedve = parms.kve / ve
    kecpdve = parms.kecp / ve
    kecpdvc = parms.kecp / vc
    kecldve = parms.kecl / ve
    kecldvc = parms.kecl / vc


    ff = np.array([ parms.VIFScale*parms.PIF,
                    np.zeros( acq.ntp ),
                    np.zeros( acq.ntp ),
                    np.zeros( acq.ntp )  ])

    # IV magnetization at each segment
    MzivSeg = ff
    MxyivSeg = ff * np.sin( np.radians(acq.FA) )

    # Initialize return variables
    Mxyev = np.zeros( (4, acq.ntp) )
    Mxyiv = np.zeros( (4, acq.ntp) )
    Mzev = np.zeros( (4, acq.ntp) )
    Mziv = np.zeros( (4, acq.ntp) )
    

    # Calculate non-trivial result
    #  Diff Eq: y'(t) = A y(t) + ff(t)
    #  Soln: y(t) = exp(A*t) * integral(0,t: exp(-A*T)*ff(T) dT)
    #  A is 4x4 matrix
    a11 = - (kvedve + kecpdve + R1Pyr + parms.extraPyrLoss)
    a13 = kecpdve
    a22 = - (kvedve + kecldve + R1Lac + parms.extraLacLoss)
    a24 = kecldve
    a31 = kecpdvc
    a33 = - (kecpdvc + parms.kpl + R1Pyr + parms.intraPyrLoss)
    a34 = parms.klp
    a42 = kecldvc
    a43 = parms.kpl
    a44 = - (kecldvc + parms.klp + R1Lac + parms.intraLacLoss)

    A = [ [a11, 0, a13, 0],
        [0, a22, 0, a24],
        [a31, 0, a33, a34],
        [0, a42, a43, a44] ]

    # Diagonalize to permit matrix integral
    dD, P = np.linalg.eig(A)
    # dD is (4,1) array containing eigenvalues
    # P is 2D array containing matrix of eigenvecters

    ### Calculate signal evolution in TR ###
    MzevSegIC = np.array(
        [parms.Pe0, parms.Le0, parms.Pi0, parms.Li0]
    ) # Set initial conditions
    for i in range( acq.ntp ):
        MxyevSeg = np.zeros( (4,) )
        MzevSeg = np.zeros( (4,) )
        
        TR = acq.TR[i]

        # First account for EV signal already present and its evolution
        #  Longitudinal magnetization available at the start of each segment
        MzevSeg = MzevSegIC
        # Signal observed at each excitation, at start of segment
        MxyevSeg = MzevSegIC * np.sin( np.radians(acq.FA[i]) )

        #At the end of this TR, Mz evolves to contribute to IC for the next segment
        if i < acq.ntp - 1: # Don't calculate after last datapoint

            MzevSeg1 = np.exp(dD*TR) * mrdivide(
                P, 
                MzevSegIC * np.cos( np.radians(acq.FA[i]) )
            ).flatten()

            # Now calculate the new spins flowing into the system
            #  Assume piecewise linear VIF. Diagonalize:
            dff1 = mrdivide(P, ff[:,i])
            dff2 = mrdivide(P, ff[:,i+1])

            # Get the slope and y-intercept for diagonalized forcing function
            b = dff1.flatten()
            m = ( (dff2-dff1)/TR ).flatten()

            # At the end of the segment inflowing spins will cause
            MzevSeg2a = (-b/dD) * ( 1-np.exp(dD*TR) )
            MzevSeg2b = m * ( (-TR/dD-((1/dD)/dD)) + np.exp(dD*TR)*((1/dD)/dD) )

            # Total signal at the end of TR equals the IC for the next TR
            MzevSegIC = np.matmul( P,(MzevSeg1 + kvedve*(MzevSeg2a+MzevSeg2b)) )


        Mxyev[:,i] = np.reshape( MxyevSeg, (4,) )
        Mzev[:, i] = np.reshape( MzevSeg, (4,) )

        Mxyiv[:, i] = MxyivSeg[:, i]
        Mziv[:, i] = MzivSeg[:, i]
    ### END OF CALCULATION LOOP ###

    Mxy = np.array([
        Mxyiv[0],  # Intravascular pyruvate
        Mxyiv[1],  # Intravascular lactate
        Mxyev[0],  # Extracellular pyruvate
        Mxyev[1],  # Extracellular lactate
        Mxyev[2],  # Intracellular pyruvate
        Mxyev[3]   # Intracellular lactate
    ])
    Mz = np.array([
        Mziv[0],  # Intravascular pyruvate
        Mziv[1],  # Intravascular lactate
        Mzev[0],  # Extracellular pyruvate
        Mzev[1],  # Extracellular lactate
        Mzev[2],  # Intracellular pyruvate
        Mzev[3]   # Intracellular lactate
    ])
    pyrSig = vb*Mz[0] + ve*Mz[2] + vc*Mz[4]
    lacSig = vb*Mz[1] + ve*Mz[3] + vc*Mz[5]
    result = sim_result(Mxy, Mz, pyrSig, lacSig)

    return result

def simp_P2L3(parms, acq):
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
            -> Mxy: np.array containing transverse magnetization for each metabolite in each compartment
                Mxy[0.:] = intravascular pyruvate
                Mxy[1,:] = intravascular lactate
                Mxy[2.:] = extravascular pyruvate
                Mxy[3,:] = extravascular lactate
            -> Mz: np.array containing longitudinal magnetization for each metabolite in each compartment
                Mz[0.:] = intravascular pyruvate
                Mz[1,:] = intravascular lactate
                Mz[2.:] = extravascular pyruvate
                Mz[3,:] = extravascular lactate
            -> pyrSig: np.array containing the total longitudinal magnetization for pyruvate as a function of time
            -> lacSig: np.array containing the total longitudinal magnetization for lactate as a function of time
    
    Note: The only initial conditions used here are pk_params.Le0 and pk_params.Pe0
          The only volume fraction used is pk_params.vb
    '''

    vb = parms.vb
    ve = (1-vb)*parms.vef
    vc = 1-vb-ve
    R1Pyr = 1 / parms.T1pyr
    R1Lac = 1 / parms.T1lac
    kvedve = parms.kve / ve
    kecpdve = parms.kecp / ve
    kecpdvc = parms.kecp / vc
    kecldve = parms.kecl / ve


    ff = np.array([ parms.VIFScale*parms.PIF,
                    np.zeros( acq.ntp ),
                    np.zeros( acq.ntp ),
                    np.zeros( acq.ntp )  ])

    # IV magnetization at each segment
    MzivSeg = ff
    MxyivSeg = ff * np.sin( np.radians(acq.FA) )

    # Initialize return variables
    Mxyev = np.zeros( (4, acq.ntp) )
    Mxyiv = np.zeros( (4, acq.ntp) )
    Mzev = np.zeros( (4, acq.ntp) )
    Mziv = np.zeros( (4, acq.ntp) )
    

    # Calculate non-trivial result
    #  Diff Eq: y'(t) = A y(t) + ff(t)
    #  Soln: y(t) = exp(A*t) * integral(0,t: exp(-A*T)*ff(T) dT)
    #  A is 4x4 matrix
    a11 = - (kecpdve + R1Pyr + parms.extraPyrLoss)
    a31 = kecpdvc
    a33 = - (kecpdvc + parms.kpl + R1Pyr + parms.intraPyrLoss)
    a43 = parms.kpl
    a44 = - (R1Lac + parms.intraLacLoss)

    A = [ [a11, 0, 0, 0],
        [0, 0, 0, 0],
        [a31, 0, a33, 0],
        [0, 0, a43, a44] ]

    # Diagonalize to permit matrix integral
    dD, P = np.linalg.eig(A)
    # dD is (4,1) array containing eigenvalues
    # P is 2D array containing matrix of eigenvecters

    # In the simplified model one of the eigenvalues will be zero, causing a divide by zero error. 
    #  To suppress this, we replace it with an arbitrarily low value
    for jj,eigen in enumerate(dD):
        if eigen==0:
            dD[jj] = 1e-20

    ### Calculate signal evolution in TR ###
    MzevSegIC = np.array(
        [parms.Pe0, parms.Le0, parms.Pi0, parms.Li0]
    ) # Set initial conditions
    for i in range( acq.ntp ):
        MxyevSeg = np.zeros( (4,) )
        MzevSeg = np.zeros( (4,) )
        
        TR = acq.TR[i]

        # First account for EV signal already present and its evolution
        #  Longitudinal magnetization available at the start of each segment
        MzevSeg = MzevSegIC
        # Signal observed at each excitation, at start of segment
        MxyevSeg = MzevSegIC * np.sin( np.radians(acq.FA[i]) )

        #At the end of this TR, Mz evolves to contribute to IC for the next segment
        if i < acq.ntp - 1: # Don't calculate after last datapoint

            MzevSeg1 = np.exp(dD*TR) * mrdivide(
                P, 
                MzevSegIC * np.cos( np.radians(acq.FA[i]) )
            ).flatten()

            # Now calculate the new spins flowing into the system
            #  Assume piecewise linear VIF. Diagonalize:
            dff1 = mrdivide(P, ff[:,i])
            dff2 = mrdivide(P, ff[:,i+1])

            # Get the slope and y-intercept for diagonalized forcing function
            b = dff1.flatten()
            m = ( (dff2-dff1)/TR ).flatten()

            # At the end of the segment inflowing spins will cause
            MzevSeg2a = (-b/dD) * ( 1-np.exp(dD*TR) )
            MzevSeg2b = m * ( (-TR/dD-((1/dD)/dD)) + np.exp(dD*TR)*((1/dD)/dD) )

            # Total signal at the end of TR equals the IC for the next TR
            MzevSegIC = np.matmul( P,(MzevSeg1 + kvedve*(MzevSeg2a+MzevSeg2b)) )


        Mxyev[:,i] = np.reshape( MxyevSeg, (4,) )
        Mzev[:, i] = np.reshape( MzevSeg, (4,) )

        Mxyiv[:, i] = MxyivSeg[:, i]
        Mziv[:, i] = MzivSeg[:, i]
    ### END OF CALCULATION LOOP ###

    Mxy = np.array([
        Mxyiv[0],  # Intravascular pyruvate
        Mxyiv[1],  # Intravascular lactate
        Mxyev[0],  # Extracellular pyruvate
        Mxyev[1],  # Extracellular lactate
        Mxyev[2],  # Intracellular pyruvate
        Mxyev[3]   # Intracellular lactate
    ])
    Mz = np.array([
        Mziv[0],  # Intravascular pyruvate
        Mziv[1],  # Intravascular lactate
        Mzev[0],  # Extracellular pyruvate
        Mzev[1],  # Extracellular lactate
        Mzev[2],  # Intracellular pyruvate
        Mzev[3]   # Intracellular lactate
    ])
    pyrSig = vb*Mz[0] + ve*Mz[2] + vc*Mz[4]
    lacSig = vb*Mz[1] + ve*Mz[3] + vc*Mz[5]
    result = sim_result(Mxy, Mz, pyrSig, lacSig)

    return result


if __name__ == '__main__':
    # Test to make sure that functions are working properly
    
    import matplotlib.pyplot as plt

    from parameters import *


    print('\nRunning each simulation with a set of dummy parameters... ')
    acq = acq_params()
    parms = pk_params(acq)

    fig, ax = plt.subplots(1,3, figsize=(12,6), num='Simulator Test')

    comp_1 = P2L1(parms, acq)
    ax[0].plot(acq.taxis, comp_1.pyrSig, color='green', label='Pyr')
    ax[0].plot(acq.taxis, comp_1.lacSig, color='blue', label='Lac')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel('Time (s)')
    ax[0].title.set_text('1 Compartment')

    comp_2 = P2L2(parms, acq)
    ax[1].plot(acq.taxis, comp_2.pyrSig, color='green', label='Pyr')
    ax[1].plot(acq.taxis, comp_2.lacSig, color='blue', label='Lac')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlabel('Time (s)')
    ax[1].title.set_text('2 Compartment')

    comp_3 = P2L3(parms, acq)
    ax[2].plot(acq.taxis, comp_3.pyrSig, color='green', label='Pyr')
    ax[2].plot(acq.taxis, comp_3.lacSig, color='blue', label='Lac')
    ax[2].legend()
    ax[2].grid()
    ax[2].set_xlabel('Time (s)')
    ax[2].title.set_text('3 Compartment')

    plt.show()

    fig, ax = plt.subplots(1,2, figsize=(12,6), num='Simp Test')

    simp = simp_P2L3(parms, acq)
    ax[0].plot(acq.taxis, comp_3.pyrSig, color='green', label='Pyr')
    ax[0].plot(acq.taxis, comp_3.lacSig, color='blue', label='Lac')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel('Time (s)')
    ax[0].set_title('Full 3 Compartment')

    ax[1].plot(acq.taxis, simp.pyrSig, color='green', label='Pyr')
    ax[1].plot(acq.taxis, simp.lacSig, color='blue', label='Lac')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlabel('Time (s)')
    ax[1].set_title('Simplified 3 Compartment')

    plt.show()

    print('\tAll functions called successfully.\n')

    is_diff = False
    for i,val in enumerate(comp_3.lacSig):
        if val != simp.lacSig[i]:
            is_diff = True

    if is_diff:
        print('These should be different and they are. Yay.')
    else:
        print("Simplified P2L3 is just a copy of P2L3 right now. Not yay.")