from __future__ import absolute_import
import tupak
import unittest
import mock
from mock import MagicMock
from mock import PropertyMock
import numpy as np
import inspect
import os
import copy


class TestGaussianLikelihood(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.sigma = 0.1
        self.x = np.linspace(0, 1, self.N)
        self.y = 2 * self.x + 1 + np.random.normal(0, self.sigma, self.N)

        def test_function(x, m, c):
            return m * x + c
        self.function = test_function

    def tearDown(self):
        del self.N
        del self.sigma
        del self.x
        del self.y
        del self.function

    def test_known_sigma(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        likelihood.parameters['m'] = 2
        likelihood.parameters['c'] = 0
        likelihood.log_likelihood()
        self.assertEqual(likelihood.sigma, self.sigma)

    def test_known_array_sigma(self):
        sigma_array = np.ones(self.N) * self.sigma
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, sigma_array)
        likelihood.parameters['m'] = 2
        likelihood.parameters['c'] = 0
        likelihood.log_likelihood()
        self.assertTrue(type(likelihood.sigma) == type(sigma_array))
        self.assertTrue(all(likelihood.sigma == sigma_array))

    def test_unknown_float_sigma(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, sigma=None)
        likelihood.parameters['m'] = 2
        likelihood.parameters['c'] = 0
        self.assertTrue(likelihood.sigma is None)
        with self.assertRaises(TypeError):
            likelihood.log_likelihood()
        likelihood.parameters['sigma'] = 1
        likelihood.log_likelihood()
        self.assertTrue(likelihood.sigma is None)

    def test_y(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        self.assertTrue(all(likelihood.y == self.y))

    def test_x(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        self.assertTrue(all(likelihood.x == self.x))

    def test_N(self):
        likelihood = tupak.core.likelihood.GaussianLikelihood(
            self.x, self.y, self.function, self.sigma)
        self.assertTrue(likelihood.N == len(self.x))


class TestGravitationalWaveTransient(unittest.TestCase):

    def setUp(self):
        start_time = 2
        duration = 3
        sampling_frequency = 4096

        self.interferometers = MagicMock()
        with mock.patch('tupak.gw.detector.InterferometerList.duration', new_callable=PropertyMock):
            self.interferometers.duration.return_value = duration
        with mock.patch('tupak.gw.detector.InterferometerList.start_time', new_callable=PropertyMock):
            self.interferometers.start_time.return_value = start_time
        with mock.patch('tupak.gw.detector.InterferometerList.sampling_frequency', new_callable=PropertyMock):
            self.interferometers.sampling_frequency.return_value = sampling_frequency
        with mock.patch('tupak.gw.detector.InterferometerList.frequency_array', new_callable=PropertyMock):
            self.interferometers.frequency_array.return_value = np.linspace(0, 9, 10)
        with mock.patch('tupak.gw.detector.InterferometerList.number_of_interferometers', new_callable=PropertyMock):
            self.interferometers.number_of_interferometers.return_value = 2
        self.waveform_generator = MagicMock()
        self.waveform_generator.duration = duration
        self.waveform_generator.sampling_frequency = sampling_frequency
        self.waveform_generator.start_time = start_time

        with mock.patch('tupak.gw.detector.InterferometerList') as m:
            m.side_effect = lambda x: x
            self.likelihood = tupak.gw.likelihood.GravitationalWaveTransient(waveform_generator=self.waveform_generator,
                                                                             interferometers=self.interferometers)

    def tearDown(self):
        del self.waveform_generator
        del self.interferometers

    def test_set_prior_none_return_dict(self):
        self.likelihood.prior = None
        self.assertTrue(type(self.likelihood.prior) == dict)

    def test_set_prior_dict(self):
        expected_dict = dict(a=1, b=2)
        self.likelihood.prior = dict(a=1, b=2)
        self.assertDictEqual(expected_dict, self.likelihood.prior)

    def test_init_with_phase_marginalization_on_without_prior(self):
        with mock.patch('tupak.core.prior.Uniform') as c:
            with mock.patch('tupak.gw.detector.InterferometerList') as m:
                c.return_value = tupak.prior.Uniform(1, 2)
                m.side_effect = lambda x: x
                self.likelihood = tupak.gw.likelihood.GravitationalWaveTransient(
                    waveform_generator=self.waveform_generator,
                    interferometers=self.interferometers,
                    phase_marginalization=True,
                    prior={})
            expected_prior = tupak.core.prior.Uniform(minimum=1, maximum=2)
            actual_prior = self.likelihood.prior['phase']
            self.assertEqual(expected_prior, actual_prior)

    def test_init_with_phase_marginalization_on_with_prior(self):
        with mock.patch('tupak.gw.detector.InterferometerList') as m:
            m.side_effect = lambda x: x
            self.likelihood = tupak.gw.likelihood.GravitationalWaveTransient(
                waveform_generator=self.waveform_generator,
                interferometers=self.interferometers,
                phase_marginalization=True,
                prior=dict(phase=tupak.prior.Uniform(minimum=100, maximum=400)))

    def test_init_with_distance_marginalization_on_without_prior(self):
        with mock.patch('tupak.core.prior.create_default_prior') as c:
            with mock.patch('tupak.gw.detector.InterferometerList') as m:
                c.return_value = tupak.prior.Uniform(1, 2)
                m.side_effect = lambda x: x
                with self.assertRaises(ValueError):
                    self.likelihood = tupak.gw.likelihood.GravitationalWaveTransient(
                       waveform_generator=self.waveform_generator,
                       interferometers=self.interferometers,
                       distance_marginalization=True,
                       prior={})


if __name__ == '__main__':
    unittest.main()
