import numpy as np
import ctypes as ct
from copy import deepcopy

so_file = "./sims.so"
sim_lib = ct.CDLL(so_file)


class c_result(ct.Structure):
    _fields_ = [
        ("shape_factor", ct.POINTER(ct.c_int)),
        ("Mxy", ct.POINTER(ct.c_double)),
        ("Mz", ct.POINTER(ct.c_double)),
        ("sep_ind", ct.POINTER(ct.c_int)),
        ("totPyrSig", ct.POINTER(ct.c_double)),
        ("totLacSig", ct.POINTER(ct.c_double))
    ]

class sim_result:
    def __init__(self, sim_out):
        print("\nStarting init of sim_result...")

        shape_fac = np.ctypeslib.as_array( sim_out.shape_factor, shape=(5,) )

        Mxy = np.ctypeslib.as_array( sim_out.Mxy, shape=(shape_fac[0],) )
        Mz = np.ctypeslib.as_array( sim_out.Mz, shape=(shape_fac[1],) )
        sep_ind = np.ctypeslib.as_array( sim_out.sep_ind, shape=(shape_fac[2],) )
        pyrSig = np.ctypeslib.as_array( sim_out.totPyrSig, shape=(shape_fac[3],) )
        lacSig = np.ctypeslib.as_array( sim_out.totLacSig, shape=(shape_fac[4],) )

        # Need to write some code that will reshape the np.arrays to play nice
        self.shape_fac = deepcopy(shape_fac) # This is just for debugging

        self.Mxy = deepcopy(Mxy)
        self.Mz = deepcopy(Mz)
        self.pyrSig = deepcopy(pyrSig)
        self.lacSig = deepcopy(lacSig)

        print("...sim_result instance success")


class acq_params(ct.Structure):
    '''
    Maybe these should have a constructor class
    so that the user doesn't have to bother with
    all of this datatype casting business
    '''
    _fields_ = [
        ("ntp", ct.c_int),
        ("FA", ct.POINTER(ct.c_double)),
        ("TR", ct.POINTER(ct.c_double))
    ]

class pk_params(ct.Structure):
    _fields_ = [
        ("kpl", ct.c_double),
        ("klp", ct.c_double),
        ("kve", ct.c_double),
        ("kecp", ct.c_double),
        ("kecl", ct.c_double),
        ("vb", ct.c_double),
        ("vef", ct.c_double),
        ("R1pyr", ct.c_double),
        ("R1lac", ct.c_double),
        ("Pi0", ct.c_double),
        ("Pe0", ct.c_double),
        ("Li0", ct.c_double),
        ("Le0", ct.c_double),
        ("PIF", ct.POINTER(ct.c_double))
    ]

sim_lib.P2L1.argtypes = [pk_params, acq_params]
sim_lib.P2L1.restype = c_result
sim_lib.free_results.argtypes = [c_result]
# sim_lib.P2L2.argtypes = [pk_params, acq_params]
# sim_lib.P2L3.argtypes = [pk_params, acq_params]

def P2L1(pk_parms, aq_parms):
    print("Calling P2L1 python function...")
    c_res = sim_lib.P2L1(pk_parms, aq_parms)
    pyt_res = sim_result(c_res)
    sim_lib.free_results(c_res)
    print("...P2L1 python func called successfully")
    return pyt_res

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import P2L1 as pyt_P2L1

    ntp = 100
    Tr = 2
    taxis = np.arange(0, Tr*ntp, Tr)

    # Initialize pk parameters object
    test_parms = pk_params()
    test_parms.kpl = 0.7
    test_parms.klp = 0.02
    test_parms.R1pyr = 1/44
    test_parms.R1lac = 1/34
    test_parms.Li0 = 0
    test_parms.PIF = (ct.c_double * ntp)(*(1000 * np.exp(-.115 * taxis))) # Arbitrary VIF here

    # Initialize acquisition parameters object
    test_acq = acq_params()
    test_acq.ntp = ntp
    test_acq.FA = (ct.c_double*ntp)(*(np.ones( (ntp,) ) * 30))  # deg
    test_acq.TR = (ct.c_double*ntp)(*(np.ones( (ntp,) ) * Tr))  # sec

    print("This is python: Everything got set up, calling C code ...\n")
    sim_res = P2L1(test_parms, test_acq)

    fdv = {}
    fdv['kpl'] = .7
    fdv['fitvarNames'] = ['kpl']
    fdv['knowns'] = {
        'T1Lac': 33, 
        'klp': .02, 
        'L0': 0
        }
    # Independent parameters
    fdv['ntp'] = ntp
    fdv['NSeg'] = 1
    fdv['TR'] = Tr
    fdv['FA'] = 30
    fdv['verbose'] = False
    # Describe acquisition scheme
    fdv['NFlips'] = (fdv['ntp']*fdv['NSeg'])
    #Describe temporal sampling scheme
    fdv['TR'] = fdv['TR'] * np.ones( (1, fdv['NFlips']) )
    fdv['taxis'] = np.cumsum(fdv['TR']) - fdv['TR'][0]
    #Describe excitation scheme
    fdv['FlipAngle'] = fdv['FA']*np.ones( (2,fdv['NFlips']) )
    fdv['data'] = [1000*np.exp(-.115*taxis)]

    pyt_Mxy, pyt_Mz = pyt_P2L1.P2L1([fdv['kpl']], fdv)

    fig, ax = plt.subplots( 1,2, figsize=(18,6), num="C Curve Test" )

    ax[0].plot(taxis, sim_res.pyrSig, color='green', label='Pyr')
    ax[0].plot(taxis, sim_res.lacSig, color='blue', label='Lac')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel('Time (s)')
    ax[0].title.set_text('C')

    ax[1].plot(taxis, pyt_Mz[0], color='green', label='Pyr')
    ax[1].plot(taxis, pyt_Mz[1], color='blue', label='Lac')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('Time (s)')
    ax[1].title.set_text('Python')

    plt.show()
    
    


