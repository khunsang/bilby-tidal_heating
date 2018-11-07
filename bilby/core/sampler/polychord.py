from __future__ import absolute_import

import numpy as np

from .base_sampler import NestedSampler
from ..utils import logger
from ..result import Result

try:
    import PyPolyChord
    from PyPolyChord.settings import PolyChordSettings
except ImportError:
    logger.debug("PyPolyChord is not installed on this system, you will not"
                 "be able to use the PolyChord Sampler")


class Polychord(NestedSampler):

    """
    Bilby wrapper of PyPolyChord
    https://arxiv.org/abs/1506.00171

    The code is currently only available on, although the author
    announced to move it to github soon(TM)
    https://ccpforge.cse.rl.ac.uk/gf/project/polychord/

    Please contact us if you can't access this code.

    Keyword arguments will be passed into `PyPolyChord.run_polychord` into the `settings`
    argument. See the PolyChord documentation for what all of those mean.
    """

    default_kwargs = dict(use_polychord_defaults=False, nlive=None, num_repeats=None,
                          nprior=-1, do_clustering=True, feedback=1, precision_criterion=0.001,
                          logzero=-1e30, max_ndead=-1, boost_posterior=0.0, posteriors=True,
                          equals=True, cluster_posteriors=True, write_resume=True,
                          write_paramnames=False, read_resume=True, write_stats=True,
                          write_live=True, write_dead=True, write_prior=True,
                          compression_factor=np.exp(-1), base_dir='polychord_chains',
                          file_root='test', seed=-1, grade_dims=None, grade_frac=None, nlives={})

    def run_sampler(self):
        import PyPolyChord
        from PyPolyChord.settings import PolyChordSettings

        if self.kwargs['use_polychord_defaults']:
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim)
        else:
            self._setup_dynamic_defaults()
            pc_kwargs = self.kwargs.copy()
            pc_kwargs.pop('use_polychord_defaults')
            settings = PolyChordSettings(nDims=self.ndim, nDerived=self.ndim, **pc_kwargs)
        self._verify_kwargs_against_default_kwargs()

        PyPolyChord.run_polychord(loglikelihood=self.log_likelihood, nDims=self.ndim,
                                  nDerived=self.ndim, settings=settings, prior=self.prior_transform)

        return Result()

    def _setup_dynamic_defaults(self):
        """ Sets up some interdependent default argument if none are given by the user """
        if not self.kwargs['grade_dims']:
            self.kwargs['grade_dims'] = [self.ndim]
        if not self.kwargs['grade_frac']:
            self.kwargs['grade_frac'] = [1.0] * len(self.kwargs['grade_dims'])
        if not self.kwargs['nlive']:
            self.kwargs['nlive'] = self.ndim * 25
        if not self.kwargs['num_repeats']:
            self.kwargs['num_repeats'] = self.ndim * 5

    def _translate_kwargs(self, kwargs):
        if 'nlive' not in kwargs:
            for equiv in self.npoints_equiv_kwargs:
                if equiv in kwargs:
                    kwargs['nlive'] = kwargs.pop(equiv)

    def log_likelihood(self, theta):
        """ Overrides the log_likelihood so that PolyChord understands it """
        return super(Polychord, self).log_likelihood(theta), theta
