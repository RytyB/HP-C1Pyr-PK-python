import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares

from .parameters import *

__all__ = ['Fit', 'Sim_Data']


# Define helper functions for Fit class
def outerErrorFunction(data, fitParmList, default_parms, acq, modelFunc):
    # This is where all the data needed for error calculation that doesn't
    #   isn't being optimized should live
    fit_vars = default_parms.write_dictionary()
    
    def innerErrorFunction(x):

        # Edit fit_vars so that the x vector hits the right parameters
        for i,val in enumerate(fitParmList.fit_vars):
            fit_vars[val] = x[i]

        test_parms = pk_params(acq, **fit_vars)

        sim = modelFunc(test_parms, acq)
        pyr_resid = (data.pyrSig - sim.pyrSig) / np.max(data.pyrSig)
        lac_resid = (data.lacSig - sim.lacSig) / np.max(data.lacSig)
        resid = np.append(pyr_resid, lac_resid)

        return resid
    return innerErrorFunction
        
def fit(fit_parms_list_and_nthfit):
    fit_parms_list, nthfit = fit_parms_list_and_nthfit
    verbose = fit_parms_list[1].verbose
    errorFunc = outerErrorFunction(*fit_parms_list)

    guess = []
    LB = []
    UB = []
    # Hard-code in initial guesses for variables
    for i,var in enumerate(fit_parms_list[1].fit_vars):
        if var == 'kpl':
            guess.append( np.random.rand()*1.5 )
            UB.append(20)
            LB.append(0)
        elif var == 'VIFScale':
            guess.append( np.random.rand()*10000 )
            UB.append(1000000)
            LB.append(0)
        elif var == 'T1pyr':
            guess.append( 43 )
            UB.append(100)
            LB.append(0)
        elif var == 'T1lac':
            guess.append( 33 )
            UB.append(100)
            LB.append(0)
        elif 'Loss' in var:
            guess.append(0)
            UB.append(100)
            LB.append(0)
        else:  # This case will mostly be rate constants
            guess.append( np.random.rand() / 10 )
            UB.append(1)
            LB.append(0)

    iter_fit = least_squares(
        fun = errorFunc,
        x0 = guess,
        bounds = (LB, UB),
        verbose = 0
    )

    opt_x = iter_fit['x']
    fitted_vars = fit_parms_list[2].write_dictionary()
    for i,var in enumerate(fit_parms_list[1].fit_vars):
        fitted_vars[var] = opt_x[i]

    acq = fit_parms_list[3]
    popt = pk_params(acq, **fitted_vars)
    score = np.linalg.norm(iter_fit['fun'])

    if verbose:
        print(f'\tFit number {nthfit + 1} complete.')

    return popt, score


class Fit:
    def __init__(self, fit_parms, default_parms, acq, data, model):

        bestScore = 10000000  # Some arbitrarily high number

        # Loop level parallelism calls each fit on a seperate process to
        #  speed up calculation
        with Pool(fit_parms.n_proc) as p:
            fit_list = [data, fit_parms, default_parms, acq, model]
            all_fits = p.map(fit, ((fit_list,n) for n in range(0, fit_parms.n_fits)))

        found_fit = False
        for i,a_fit in enumerate(all_fits):
            if a_fit[1] < bestScore:
                best_i = i
                bestScore = a_fit[1]
                found_fit = True

        if not found_fit:
            # This condition should never realistically be run
            print('Did not find a fit that exceeded quality threshold.')
            return
        
        self.popt = all_fits[best_i][0]  # pk_params object that produces best fit
        opt_curves = model(self.popt, acq)
        self.Mxy = opt_curves.Mxy
        self.Mz = opt_curves.Mz
        self.pyrSig = opt_curves.pyrSig
        self.lacSig = opt_curves.lacSig

        if fit_parms.verbose:
            self.popt.pretty_print('Fitted PK Parms')

        return
    def calculate_NMSE(self):
        return
    def calculate_MSE(self):
        pyr_resid = self.pyrSig - self.data.pyrSig
        lac_resid = self.lacSig - self.data.lacSig
        pyr_MSE = (np.linalg.norm(pyr_resid)**2) / len(pyr_resid)
        lac_MSE = (np.linalg.norm(lac_resid)**2) / len(lac_resid)
        tot_MSE = (np.linalg.norm(pyr_resid)**2 + np.linalg.norm(lac_resid)**2) / (len(pyr_resid) + len(lac_resid))
        return (pyr_MSE, lac_MSE, tot_MSE)
    def calculate_RSS(self):
        pyr_resid = self.pyrSig - self.data.pyrSig
        lac_resid = self.lacSig - self.data.lacSig
        pyr_RSS = np.linalg.norm(pyr_resid) ** 2
        lac_RSS = np.linalg.norm(lac_resid) ** 2
        tot_RSS = pyr_RSS + lac_RSS
        return (pyr_RSS, lac_RSS, tot_RSS)
    

class Sim_Data:
    def __init__(self, parms, acq, model):

        self.taxis = np.cumsum( acq.TR ) - acq.TR[0]
        try:
            sim = model(parms, acq)
            self.pyrSig = sim.pyrSig
            self.lacSig = sim.lacSig
        except:
            print('\nError: Signal curve could not be calculated. Model function may not be valid.\n')

        return
    def noisify(self, peakSnr):

        # Calculate the variance of the distribution from peakSNR
        if np.max(self.pyrSig) > np.max(self.lacSig):
            sigma = np.max(self.pyrSig) / peakSnr
        else:
            sigma = np.max(self.lacSig) / peakSnr

        # Sample Gaussian and add to curve at every time point
        for i,_ in enumerate(self.pyrSig):
            self.pyrSig[i] += np.random.normal(loc=0, scale=sigma)
            self.lacSig[i] += np.random.normal(loc=0, scale=sigma) 

        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from simulators import *

    vars_to_fit = ['kpl', 'VIFScale']

    fit_parms = fit_params(fit_vars=vars_to_fit, n_proc=4, n_fits=20, verbose=True)
    acq = acq_params()
    parms = pk_params(acq)
    parms.pretty_print('Sim_Data PK Parms')

    print(fit_parms.fit_vars)

    test_data = Sim_Data(parms, acq, P2L3)
    test_fit = Fit(fit_parms, parms, acq, test_data, P2L3)
    test_sim = P2L3(parms, acq)


    fig, ax = plt.subplots(figsize=(12,6), num='Sim_Data and Fit Test')
    ax.plot(test_data.taxis, test_fit.pyrSig, color='green', label='Pyr')
    ax.plot(test_data.taxis, test_fit.lacSig, color='blue', label='Lac') 
    test_data.noisify(peakSnr=20)
    ax.scatter(test_data.taxis, test_data.pyrSig, marker='x', color='green', label='Pyr')
    ax.scatter(test_data.taxis, test_data.lacSig, marker='x', color='blue', label='Lac')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.title.set_text('P2L3 Sim Fitted to Noisy Data')
    plt.show()

