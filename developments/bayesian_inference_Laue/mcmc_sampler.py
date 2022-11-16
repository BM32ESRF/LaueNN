# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:35:30 2021

@author: PURUSHOT
Image  plot for Laue article 1

"""

from copy import copy
import numpy as np
import os
import pickle
import pymc
import math
# =============================================================================
class MCMCSampler:
    '''
    Class for MCMC sampling; based on PyMC (https://github.com/pymc-devs/pymc).
    Uses Bayesian inference, MCMC, and a model to estimate parameters with
    quantified uncertainty based on a set of observations. 
    '''

    def __init__(self, data, model, params, working_dir='./',
                 storage_backend='pickle'):
        '''
        Set data and model class member variables, set working directory,
        and choose storage backend.

        :param data: data to use to inform MCMC parameter estimation;
            should be same type/shape/size as output of model.
        :type data: array_like
        :param model: model for which parameters are being estimated via MCMC;
            should return output in same type/shape/size as data. A baseclass
            exists in the Model module that is recommended to define the model
            object; i.e., model.__bases__ == <class model.Model.Model>,)
        :type model: object
        :param params: map where keys are the unknown parameter 
            names (string) and values are lists that define the prior
            distribution of the parameter [dist name, dist. arg #1, dist. arg
            #2, etc.]. The distribution arguments are defined in the PyMC
            documentation: https://pymc-devs.github.io/pymc/.
        :type params: dict
        :storage_backend: determines which format to store mcmc data,
            see self.avail_backends for a list of options.
        :type storage_backend: str
        '''
        self.params = params
        self.model = model
        self.working_dir = working_dir
        self.sigma = 1
        # Check for valid storage backend
        self.avail_backends = ['pickle', 'ram', 'no_trace', 'txt', 'sqlite',
                               'hdf5']
        if storage_backend not in self.avail_backends:
            msg = 'invalid backend option for storage: %s.' % storage_backend
            raise NameError(msg)
        else:
            self.storage_backend = storage_backend
            if storage_backend == 'pickle':
                self.loader = pymc.database.pickle 
                self.backend_ext = '.p'
            elif storage_backend == 'ram':
                self.loader = pymc.database.ram 
            elif storage_backend == 'no_trace':
                self.loader = pymc.database.no_trace 
            elif storage_backend == 'txt':
                self.loader = pymc.database.txt
                self.backend_ext = '.txt'
            elif storage_backend == 'sqlite':
                self.loader = pymc.database.sqlite
                self.backend_ext = '.db'
            elif storage_backend == 'hdf5':
                self.loader = pymc.database.hdf5
                self.backend_ext = '.h5'
        # verify working dir exists
        self._verify_working_dir()
        # Number of data points and parameters, respectively
        self.n = len(data)
        self.p = len(params)
        # Verify data format - for now, must be a list of crack lengths
        self.data = self._verify_data_format(data)
        # Initialize pymc model and MCMC database
        self.pymc_mod = None
        self.db = None
        self._initialize_plotting()

    def pymcplot(self):
        '''
        Generates a pymc plot for each parameter in self.. This plot
        includes a trace, histogram, and autocorrelation plot. For more control
        over the plots, see MCMCplots module. This is meant as a diagnostic
        tool only.
        '''
        from pymc.Matplot import plot as pymc_plot
        for name in self.params.keys():
            pymc_plot(self.db.trace(name), format='png', path=self.working_dir)
        print( 'pymc plots generated for  = %s' % self.params.keys())


    def sample(self, num_samples, burnin, step_method='adaptive', interval=1000,
               delay=0, tune_throughout=False, scales=None, cov=None, thin=1,
               phi=None, burn_till_tuned=False, proposal_distribution ='normal', verbose=0):
        '''
        Initiates MCMC sampling of posterior distribution using the
        model defined using the generate_pymc_model method. Sampling is
        conducted using the PyMC module. Parameters are as follows:
        
            - num_samples : number of samples to draw (int)
            - burnin : number of samples for burn-in (int)
            - adaptive : toggles adaptive metropolis sampling (bool)
            - step_method : step method for sampling; options are:
                  o adaptive - regular adaptive metropolis
                  o DRAM - delayed rejection adaptive metropolis
                  o metropolis - standard metropolis aglorithm
            - interval : defines frequency of covariance updates (only
                  applicable to adaptive methods)
            - delay : how long before first cov update occurs (only applicable
                  to adaptive methods)
            - tune_throughout : True > tune proposal covariance even after
                  burnin, else only tune proposal covariance during burn in
            - scales : scale factors for the diagonal of the multivariate
                  normal proposal distribution; must be dictionary with
                  keys = .keys() and values = scale for that param.
            - phi : cooling step; only used for SMC sampler
            -proposal_distribution ='normal' or 'prior'
        '''
        # Check if pymc_mod is defined
        if self.pymc_mod == None:
            raise NameError('Cannot sample; self.pymc_mod not defined!')

        # Setup pymc sampler (set to output results as pickle file
        if self.storage_backend != 'ram':
            dbname = self.working_dir+'/mcmc'+self.backend_ext
            if self.storage_backend == 'hdf5':
                self._remove_hdf5_if_exists(dbname)
        else:
            dbname = None
        self.MCMC = pymc.MCMC(self.pymc_mod, db=self.storage_backend, 
                              dbname=dbname, verbose=verbose)

        # Set  as random variables (stochastics)
        parameter_RVs = [self.pymc_mod[i] for i in range(self.p)]
        # Set scales
        if scales != None:
            if len(scales) != len(parameter_RVs):
                err_msg = 'len(scales) must be equal to num '
                raise ValueError(err_msg)
            scales = {rv: scales[rv.__name__] for rv in parameter_RVs}
        else:
            scales = None
        # assign step method
        if step_method.lower() == 'adaptive':
            print( 'adaptation delay = %s' % delay)
            print( 'adaptation interval = %s' % interval)
            self.MCMC.use_step_method(pymc.AdaptiveMetropolis, parameter_RVs,
                              shrink_if_necessary=True, interval=interval,
                              delay=delay, scales=scales, cov=cov,
                              verbose=verbose)

        elif step_method.lower() == 'metropolis':
            # if no scales provided,
            if scales is None:
                scales = dict()
                for rv in parameter_RVs:
                    scales[rv] = abs(rv.value/500.)
            # set Metropolis step method for each random variable
            for RV, scale in scales.items():
                self.MCMC.use_step_method(pymc.Metropolis, RV, scale=scale,
                                          proposal_distribution=proposal_distribution, verbose=verbose)

        else:
            raise KeyError('Unknown step method passed to sample()')

        # sample
        if tune_throughout:
            print( 'tuning proposal distribution throughout mcmc')
        self.MCMC.sample(num_samples, burnin, tune_throughout=tune_throughout,
                         thin=thin, burn_till_tuned=burn_till_tuned, verbose=verbose)
        #self.MCMC.db.close() TODO
        self._enable_trace_plotting()
        self.MCMC.db.commit()

    @staticmethod
    def _remove_hdf5_if_exists(dbname):
        if os.path.exists(dbname):
            print( 'database %s exists; overwriting...' % dbname)
            os.remove(dbname)
        return None
    
    def psi_i_logp(self, b=None, c=None, alpha=None,
                  beta=None, gamma=None, angx=None, angy=None, angz=None):
        """
        Evalute the log-likelihood of a given data
        """
        try:
            #sim_set, exp_set = self.model.evaluate1(b,c,alpha,beta,gamma,angx,angy,angz)
            #distanceterm = np.sqrt((X - pixX) ** 2 + (Y - pixY) ** 2)
            #like0 = -0.91894 - 0.5*sum((exp_set-sim_set)**2)
            #return like0
            
            distance, n = self.model.evaluate1(b,c,alpha,beta,gamma,angx,angy,angz)
            sigma = self.sigma
            ll = - 0.5*sum(distance/sigma**2)
            return ll
        
        except ValueError:
            print("Error")
            return  float(-1.7e+100)
        
    def generate_pymc_model(self, q0=None, ssq0=None, std_dev0=1.0):
        '''
        PyMC stochastic model generator that uses the parameter dictionary,
        self. and optional inputs:
            
            - q0        : a dictionary of initial values corresponding to keys 
                          in
            - std_dev0  : an estimate of the initial standard deviation
            - ssq0      : the sum of squares error using q0 and self.data. Only
                          used if initial var, var0, is None.
        '''
        # Set up pymc objects from self.
        self.sigma = std_dev0
        parents, pymc_mod, pymc_mod_order = self.generate_pymc_(self, q0)
        precision = 1./std_dev0**2
        pymc_mod_addon = []
        pymc_mod_order_addon = []
        # Define deterministic model
        #model = pymc.Deterministic(self.model.evaluate, name='model_RV', 
        #                           doc='model_RV', parents=parents, trace=True, plot=True)
        # Posterior (random variable, normally distributed, centered at model)
        #results = pymc.Normal('results', mu=model, tau=precision,
        #                      value=self.data, observed=True)
        # Assemble model and return
        #pymc_mod += [model, results] + pymc_mod_addon # var is last
        #pymc_mod_order += ['model', 'results'] + pymc_mod_order_addon 
        #self.pymc_mod = pymc_mod
        #self.pymc_mod_order = pymc_mod_order
        
        results = pymc.Potential(logp = self.psi_i_logp, name = 'results', doc =  'results',
                                parents = parents, verbose = 0, cache_depth = 2)
        # Assemble model and return
        pymc_mod += [results]
        pymc_mod_order += ['results']
        self.pymc_mod = pymc_mod
        self.pymc_mod_order = pymc_mod_order

    def generate_pymc_(self, params, q0=None):
        '''
        Creates PyMC objects for each param in  dictionary

        NOTE: the second argument for normal distributions is VARIANCE

        Prior option:
            An arbitrary prior distribution derived from a set of samples (e.g.,
            a previous mcmc run) can be passed with the following syntax:

                 = {<name> : ['KDE', <pymc_database>, <param_names>]}

            where <name> is the name of the distribution (e.g., 'prior' or
            'joint_dist'), <pymc_database> is the pymc database containing the
            samples from which the prior distribution will be estimated, and
            <param_names> are the children parameter names corresponding to the
            dimension of the desired sample array. This method will use all
            samples of the Markov chain contained in <pymc_database> for all
            traces named in <param_names>. Gaussian kernel-density estimation
            is used to derive the joint parameter distribution, which is then
            treated as a prior in subsequent mcmc analyses using the current
            class instance. The parameters named in <param_names> will be
            traced as will the multivariate distribution named <name>.
        '''
        pymc_mod = []
        pymc_mod_order = []
        parents = dict()
        # Iterate through , assign prior distributions
        for key, args in sorted(self.params.items(), reverse=True): ##reverse to have C12 first
            # Distribution name should be first entry in [key]
            print("additon order: "+key)
            dist = args[0].lower()
            
            if dist == 'normal':
                if q0 == None:
                    RV = [pymc.Normal(key, mu=args[1], tau=1./args[2])]
                else:
                    RV = [pymc.Normal(key, mu=args[1], tau=1./args[2],value=q0[key])]
            
            elif dist == 'uniform': ### modify for our case i.e. BORN STABILITY
                if q0 == None:
                    RV = [pymc.Uniform(key, lower=args[1], upper=args[2])]
                else:
                    RV = [pymc.Uniform(key, lower=args[1], upper=args[2], value=q0[key])]
            else:
                raise KeyError('The distribution "'+dist+'" is not supported.')

            parents[key] = RV[0]
            pymc_mod_order.append(key)
            pymc_mod += RV
        
        return parents, pymc_mod, pymc_mod_order


    def save_model(self, fname='model.p'):
        '''
        Saves model in pickle file with name working_dir + fname.
        '''
        # store model
        model = {'model':self.model}

        # dump
        with open(self.working_dir+'/'+fname, 'w') as f:
            pickle.dump(model, f)


    def _verify_working_dir(self):
        '''
        Ensure specified working directory exists.
        '''
        if not os.path.isdir(self.working_dir):
            print( 'Working directory does not exist; creating...')
            os.mkdir(self.working_dir)
            self.working_dir = os.path.realpath(self.working_dir)


    def _verify_data_format(self, data):
        '''
        Ensures that data is a single list.
        '''
        # For now, data should be a list of floats (e.g., of crack lengths)
        if type(data) == list or type(data) == np.ndarray:
            return np.array(data)
        else:
            raise TypeError('Data must be a single list of floats.')

    def _initialize_plotting(self):
        self.plot_pairwise = self._raise_missing_trace_error
        self.plot_pdf = self._raise_missing_trace_error
        # self.plot_residuals = residuals
        # self.plot_data = time_vs_observations
        return None

    @staticmethod
    def _raise_missing_trace_error(*args, **kwargs):
        raise ValueError('no trace to plot; use sample() first')


    def _enable_trace_plotting(self):
        # self.plot_pairwise = functools.partial(pairwise, trace=trace)
        # self.plot_pdf = functools.partial(pdf, trace=trace)
        return None