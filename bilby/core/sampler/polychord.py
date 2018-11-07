from __future__ import absolute_import

import numpy as np
import PyPolyChord
from PyPolyChord.settings import PolyChordSettings

from .base_sampler import NestedSampler
from ..utils import logger

# from PyPolyChord.priors import UniformPrior


class Polychord(NestedSampler):
    default_kwargs = dict(use_polychord_defaults=False, nlive=None, num_repeats=None,
                          nprior=-1, do_clustering=True, feedback=1, precision_criterion=0.001,
                          logzero=-1e30, max_ndead=-1, boost_posterior=0.0, posteriors=True,
                          equals=True, cluster_posteriors=True, write_resume=True,
                          write_paramnames=False, read_resume=True, write_stats=True,
                          write_live=True, write_dead=True, write_prior=True,
                          compression_factor=np.exp(-1), base_dir='polychord_chains',
                          file_root='test', seed=-1, grade_dims=None, grade_frac=None, nlives={})

    def run_sampler(self):
        if self.kwargs['use_polychord_defaults']:
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim)
        else:
            self._setup_dynamic_defaults()
            pc_kwargs = self.kwargs.copy()
            pc_kwargs.pop('use_polychord_defaults')
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim, **pc_kwargs)

        self._verify_kwargs_against_default_kwargs()
        PyPolyChord.run_polychord(loglikelihood=self.log_likelihood_wrapper, nDims=self.ndim,
                                  nDerived=self.ndim, settings=settings, prior=self.prior_transform)

        return None

    def _setup_dynamic_defaults(self):
        if not self.kwargs['grade_dims']:
            self.kwargs['grade_dims'] = [self.ndim]
        if not self.kwargs['grade_frac']:
            self.kwargs['grade_frac'] = [1.0] * len(self.kwargs['grade_dims'])
        if not self.kwargs['nlive']:
            self.kwargs['nlive'] = self.ndim * 25
        if not self.kwargs['num_repeats']:
            self.kwargs['num_repeats'] = self.ndim * 25

    def _verify_kwargs_against_default_kwargs(self):
        """
        Check if the kwargs are contained in the list of available arguments
        of the external sampler.
        """
        args = self.default_kwargs

        bad_keys = []
        for user_input in self.kwargs.keys():
            if user_input not in args:
                logger.warning(
                    "Supplied argument '{}' not an argument of '{}', removing."
                    .format(user_input, self.__class__.__name__))
                bad_keys.append(user_input)
        for key in bad_keys:
            self.kwargs.pop(key)

    def log_likelihood_wrapper(self, theta):
        return self.log_likelihood(theta), theta
