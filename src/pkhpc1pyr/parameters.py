from numpy import cumsum
from numpy import ones


from scipy.stats import gamma


__all__ = [
    'fit_params',
    'pk_params',
    'acq_params',
    'sim_result'
]


class fit_params:
    def __init__(self, **kwargs):
        self.n_proc = kwargs.get('n_proc', 1)
        self.n_fits = kwargs.get('n_fits', 1)
        self.fit_vars = kwargs.get('fit_vars', ['kpl'])
        self.verbose = kwargs.get('verbose', False)

class acq_params:
    def __init__(self, **kwargs):
        self.ntp = kwargs.get('ntp', 100)
        self.FA = kwargs.get('FA', 20*ones( (self.ntp,) ))
        self.TR = kwargs.get('TR', 2*ones( (self.ntp,) ))
        self.taxis = cumsum(self.TR) - self.TR[0]
        return

class pk_params:
    def __init__(self, acq_parms, **kwargs):
        def_pif = gamma.pdf(a=2.8, scale=4.5, x=acq_parms.taxis)
        self.kpl = kwargs.get('kpl', .5)
        self.VIFScale = kwargs.get('VIFScale', 1000)
        self.klp = kwargs.get('klp', .02)
        self.kve = kwargs.get('kve', .02)
        self.kecp = kwargs.get('kecp', .02)
        self.kecl = kwargs.get('kecl', self.kecp)
        self.vb = kwargs.get('vb', .09)
        self.vef = kwargs.get('vef', .5)
        self.T1pyr = kwargs.get('T1pyr', 43)
        self.T1lac = kwargs.get('T1lac', 33)
        self.Pi0 = kwargs.get('Pi0', 0)
        self.Pe0 = kwargs.get('Pe0', 0)
        self.Li0 = kwargs.get('Li0', 0)
        self.Le0 = kwargs.get('Le0', 0)
        self.PIF = kwargs.get('PIF', def_pif)
        self.intraPyrLoss = kwargs.get('intraPyrLoss', 0)
        self.intraLacLoss = kwargs.get('intraLacLoss', 0)
        self.extraPyrLoss = kwargs.get('extraPyrLoss', 0)
        self.extraLacLoss = kwargs.get('extraLacLoss', 0)
        return
    def write_dictionary(self):
        my_dict = {
            'kpl':self.kpl,
            'VIFScale':self.VIFScale,
            'klp':self.klp,
            'kve':self.kve,
            'kecp':self.kecp,
            'kecl':self.kecl,
            'vb':self.vb,
            'vef':self.vef,
            'T1pyr':self.T1pyr,
            'T1lac':self.T1lac,
            'Pi0':self.Pi0,
            'Pe0':self.Pe0,
            'Li0':self.Li0,
            'Le0':self.Le0,
            'PIF':self.PIF,
            'intraPyrLoss':self.intraPyrLoss,
            'intraLacLoss':self.intraLacLoss,
            'extraPyrLoss':self.extraPyrLoss,
            'extraLacLoss':self.extraLacLoss
        }
        return my_dict
    def pretty_print(self, title='Pharmacokinetic Parameters'):

        print('\n--- ' + title + ' ---')
        print('kpl = ' + str(self.kpl))
        print('VIFScale = ' + str(self.VIFScale))
        print('klp = ' + str(self.klp))
        print('kve = ' + str(self.kve))
        print('kecp = ' + str(self.kecp))
        print('kecl = ' + str(self.kecl))
        print('vb = ' + str(self.vb))
        print('vef = ' + str(self.vef))
        print('T1pyr = ' + str(self.T1pyr))
        print('T1lac = ' + str(self.T1lac))
        print('intraPyrLoss = ' + str(self.intraPyrLoss))
        print('intraLacLoss = ' + str(self.intraLacLoss))
        print('extraPyrLoss = ' + str(self.extraLacLoss))
        print('extraLacLoss = ' + str(self.extraLacLoss))
        print()

        return

class sim_result:
    def __init__(self, Mxy, Mz, pyrSig, lacSig):
        self.Mxy = Mxy
        self.Mz = Mz
        self.pyrSig = pyrSig
        self.lacSig = lacSig
        return


if __name__ == '__main__':
    print('\nInitializing every class with default values...')
    fit_params()
    acq = acq_params()
    pk_params(acq)
    print('\tAll classes successfully instantiated.')

    print('Included classes are as follows:')
    print(__all__)
    print()
