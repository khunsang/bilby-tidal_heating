from __future__ import division, absolute_import
import unittest
import bilby
import numpy as np


class TestBasicGWTransient(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
            dec=-1.2108)
        self.interferometers = bilby.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=2048, duration=4)
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=4, sampling_frequency=2048,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            )

        self.likelihood = bilby.gw.likelihood.BasicGravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator
        )
        self.likelihood.parameters = self.parameters.copy()

    def tearDown(self):
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.likelihood

    def test_noise_log_likelihood(self):
        """Test noise log likelihood matches precomputed value"""
        self.likelihood.noise_log_likelihood()
        self.assertAlmostEqual(-4037.0994372143414, self.likelihood.noise_log_likelihood(), 3)

    def test_log_likelihood(self):
        """Test log likelihood matches precomputed value"""
        self.likelihood.log_likelihood()
        self.assertAlmostEqual(self.likelihood.log_likelihood(),
                               -4054.2229111227016, 3)

    def test_log_likelihood_ratio(self):
        """Test log likelihood ratio returns the correct value"""
        self.assertAlmostEqual(
            self.likelihood.log_likelihood()
            - self.likelihood.noise_log_likelihood(),
            self.likelihood.log_likelihood_ratio(), 3)

    def test_likelihood_zero_when_waveform_is_none(self):
        """Test log likelihood returns np.nan_to_num(-np.inf) when the
        waveform is None"""
        self.likelihood.parameters['mass_2'] = 32
        self.assertEqual(self.likelihood.log_likelihood_ratio(),
                         np.nan_to_num(-np.inf))
        self.likelihood.parameters['mass_2'] = 29

    def test_repr(self):
        expected = 'BasicGravitationalWaveTransient(interferometers={},\n\twaveform_generator={})'.format(
            self.interferometers, self.waveform_generator)
        self.assertEqual(expected, repr(self.likelihood))


class TestGWTransient(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
            dec=-1.2108)
        self.interferometers = bilby.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration)
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )

        self.prior = bilby.gw.prior.BBHPriorDict()
        self.prior['geocent_time'] = bilby.prior.Uniform(
            minimum=self.parameters['geocent_time'] - self.duration / 2,
            maximum=self.parameters['geocent_time'] + self.duration / 2)

        self.likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator, priors=self.prior.copy()
        )
        self.likelihood.parameters = self.parameters.copy()

    def tearDown(self):
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.prior
        del self.likelihood

    def test_noise_log_likelihood(self):
        """Test noise log likelihood matches precomputed value"""
        self.likelihood.noise_log_likelihood()
        self.assertAlmostEqual(-4037.0994372143414, self.likelihood.noise_log_likelihood(), 3)

    def test_log_likelihood(self):
        """Test log likelihood matches precomputed value"""
        self.likelihood.log_likelihood()
        self.assertAlmostEqual(self.likelihood.log_likelihood(),
                               -4054.2229111227016, 3)

    def test_log_likelihood_ratio(self):
        """Test log likelihood ratio returns the correct value"""
        self.assertAlmostEqual(
            self.likelihood.log_likelihood()
            - self.likelihood.noise_log_likelihood(),
            self.likelihood.log_likelihood_ratio(), 3)

    def test_likelihood_zero_when_waveform_is_none(self):
        """Test log likelihood returns np.nan_to_num(-np.inf) when the
        waveform is None"""
        self.likelihood.parameters['mass_2'] = 32
        self.assertEqual(self.likelihood.log_likelihood_ratio(),
                         np.nan_to_num(-np.inf))
        self.likelihood.parameters['mass_2'] = 29

    def test_repr(self):
        expected = 'GravitationalWaveTransient(interferometers={},\n\twaveform_generator={},\n\t' \
                   'time_marginalization={}, distance_marginalization={}, phase_marginalization={}, ' \
                   'priors={})'.format(self.interferometers, self.waveform_generator, False, False, False, self.prior)
        self.assertEqual(expected, repr(self.likelihood))


class TestTimeMarginalization(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259640, ra=1.375,
            dec=-1.2108)

        self.interferometers = bilby.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=1126259640)

        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            start_time=1126259640)

        self.prior = bilby.gw.prior.BBHPriorDict()

        self.likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator, priors=self.prior.copy()
        )

        self.likelihood.parameters = self.parameters.copy()

    def tearDown(self):
        del self.duration
        del self.sampling_frequency
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.prior
        del self.likelihood

    def test_time_marginalisation_full_segment(self):
        """
        Test time marginalised likelihood matches brute force version over the
        whole segment.
        """
        likes = []
        lls = []
        self.prior['geocent_time'] = bilby.prior.Uniform(
            minimum=self.waveform_generator.start_time,
            maximum=self.waveform_generator.start_time + self.duration)
        self.time = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            time_marginalization=True, priors=self.prior.copy()
        )
        times = self.waveform_generator.start_time + np.linspace(
            0, self.duration, 4097)[:-1]
        for time in times:
            self.likelihood.parameters['geocent_time'] = time
            lls.append(self.likelihood.log_likelihood_ratio())
            likes.append(np.exp(lls[-1]))

        marg_like = np.log(np.trapz(
            likes * self.prior['geocent_time'].prob(times), times))
        self.time.parameters = self.parameters.copy()
        self.time.parameters['geocent_time'] = self.waveform_generator.start_time
        self.assertAlmostEqual(marg_like, self.time.log_likelihood_ratio(),
                               delta=0.5)

    def test_time_marginalisation_partial_segment(self):
        """
        Test time marginalised likelihood matches brute force version over the
        whole segment.
        """
        likes = []
        lls = []
        self.prior['geocent_time'] = bilby.prior.Uniform(
            minimum=self.parameters['geocent_time'] + 1 - 0.1,
            maximum=self.parameters['geocent_time'] + 1 + 0.1)
        self.time = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            time_marginalization=True, priors=self.prior.copy()
        )
        times = self.waveform_generator.start_time + np.linspace(
            0, self.duration, 4097)[:-1]
        for time in times:
            self.likelihood.parameters['geocent_time'] = time
            lls.append(self.likelihood.log_likelihood_ratio())
            likes.append(np.exp(lls[-1]))

        marg_like = np.log(np.trapz(
            likes * self.prior['geocent_time'].prob(times), times))
        self.time.parameters = self.parameters.copy()
        self.time.parameters['geocent_time'] = self.waveform_generator.start_time
        self.assertAlmostEqual(marg_like, self.time.log_likelihood_ratio(),
                               delta=0.5)


class TestMarginalizedLikelihood(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
            dec=-1.2108)

        self.interferometers = bilby.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=self.parameters['geocent_time'] - self.duration / 2)

        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )

        self.prior = bilby.gw.prior.BBHPriorDict()
        self.prior['geocent_time'] = bilby.prior.Uniform(
            minimum=self.parameters['geocent_time'] - self.duration / 2,
            maximum=self.parameters['geocent_time'] + self.duration / 2)

    def test_cannot_instantiate_marginalised_likelihood_without_prior(self):
        self.assertRaises(
            ValueError,
            lambda: bilby.gw.likelihood.GravitationalWaveTransient(
                interferometers=self.interferometers,
                waveform_generator=self.waveform_generator,
                phase_marginalization=True))

    def test_generating_default_time_prior(self):
        temp = self.prior.pop('geocent_time')
        new_prior = self.prior.copy()
        like = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator, priors=new_prior,
            time_marginalization=True
        )
        same = all([temp.minimum == like.priors['geocent_time'].minimum,
                    temp.maximum == like.priors['geocent_time'].maximum,
                    new_prior['geocent_time'] == temp.minimum])
        self.assertTrue(same)
        self.prior['geocent_time'] = temp

    def test_generating_default_phase_prior(self):
        temp = self.prior.pop('phase')
        new_prior = self.prior.copy()
        like = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator, priors=new_prior,
            phase_marginalization=True
        )
        same = all([temp.minimum == like.priors['phase'].minimum,
                    temp.maximum == like.priors['phase'].maximum,
                    new_prior['phase'] == float(0)])
        self.assertTrue(same)
        self.prior['phase'] = temp


class TestPhaseMarginalization(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
            dec=-1.2108)

        self.interferometers = bilby.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration)

        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        )

        self.prior = bilby.gw.prior.BBHPriorDict()
        self.prior['geocent_time'] = bilby.prior.Uniform(
            minimum=self.parameters['geocent_time'] - self.duration / 2,
            maximum=self.parameters['geocent_time'] + self.duration / 2)

        self.likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator, priors=self.prior.copy()
        )

        self.phase = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            phase_marginalization=True, priors=self.prior.copy()
        )
        for like in [self.likelihood, self.phase]:
            like.parameters = self.parameters.copy()

    def tearDown(self):
        del self.duration
        del self.sampling_frequency
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.prior
        del self.likelihood
        del self.phase

    def test_phase_marginalisation(self):
        """Test phase marginalised likelihood matches brute force version"""
        like = []
        phases = np.linspace(0, 2 * np.pi, 1000)
        for phase in phases:
            self.likelihood.parameters['phase'] = phase
            like.append(np.exp(self.likelihood.log_likelihood_ratio()))

        marg_like = np.log(np.trapz(like, phases) / (2 * np.pi))
        self.phase.parameters = self.parameters.copy()
        self.assertAlmostEqual(marg_like, self.phase.log_likelihood_ratio(),
                               delta=0.5)


class TestTimePhaseMarginalization(unittest.TestCase):

    def setUp(self):
        np.random.seed(500)
        self.duration = 4
        self.sampling_frequency = 2048
        self.parameters = dict(
            mass_1=31., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.0, tilt_2=0.0,
            phi_12=1.7, phi_jl=0.3, luminosity_distance=4000., iota=0.4,
            psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375,
            dec=-1.2108)

        self.interferometers = bilby.gw.detector.InterferometerList(['H1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=1126259640)

        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            start_time=1126259640)

        self.prior = bilby.gw.prior.BBHPriorDict()
        self.prior['geocent_time'] = bilby.prior.Uniform(
            minimum=self.parameters['geocent_time'] - self.duration / 2,
            maximum=self.parameters['geocent_time'] + self.duration / 2)

        self.likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator, priors=self.prior.copy()
        )

        self.time = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            time_marginalization=True, priors=self.prior.copy()
        )

        self.phase = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            phase_marginalization=True, priors=self.prior.copy()
        )

        self.time_phase = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=self.interferometers,
            waveform_generator=self.waveform_generator,
            time_marginalization=True, phase_marginalization=True,
            priors=self.prior.copy()
        )
        for like in [self.likelihood, self.time, self.phase, self.time_phase]:
            like.parameters = self.parameters.copy()

    def tearDown(self):
        del self.duration
        del self.sampling_frequency
        del self.parameters
        del self.interferometers
        del self.waveform_generator
        del self.prior
        del self.likelihood
        del self.time
        del self.phase
        del self.time_phase

    def test_time_phase_marginalisation(self):
        """Test time and marginalised likelihood matches brute force version"""
        like = []
        times = np.linspace(self.prior['geocent_time'].minimum,
                            self.prior['geocent_time'].maximum, 4097)[:-1]
        for time in times:
            self.phase.parameters['geocent_time'] = time
            like.append(np.exp(self.phase.log_likelihood_ratio()))

        marg_like = np.log(np.trapz(like, times)
                           / self.waveform_generator.duration)
        self.time_phase.parameters = self.parameters.copy()
        self.assertAlmostEqual(marg_like,
                               self.time_phase.log_likelihood_ratio(),
                               delta=0.5)

        like = []
        phases = np.linspace(0, 2 * np.pi, 1000)
        for phase in phases:
            self.time.parameters['phase'] = phase
            like.append(np.exp(self.time.log_likelihood_ratio()))

        marg_like = np.log(np.trapz(like, phases) / (2 * np.pi))
        self.time_phase.parameters = self.parameters.copy()
        self.assertAlmostEqual(marg_like,
                               self.time_phase.log_likelihood_ratio(),
                               delta=0.5)


class TestROQLikelihood(unittest.TestCase):

    def setUp(self):
        self.duration = 4
        self.sampling_frequency = 2048

        roq_dir = '/root/roq_basis'

        basis_matrix_linear = np.load("{}/B_linear.npy".format(roq_dir)).T
        freq_nodes_linear = np.load("{}/fnodes_linear.npy".format(roq_dir))

        basic_matrix_quadratic = np.load("{}/B_quadratic.npy".format(roq_dir)).T
        freq_nodes_quadratic = np.load("{}/fnodes_quadratic.npy".format(roq_dir))

        self.test_parameters = dict(
            mass_1=36.0, mass_2=36.0, a_1=0.0, a_2=0.0, tilt_1=0.0,
            tilt_2=0.0, phi_12=1.7, phi_jl=0.3, luminosity_distance=5000.,
            iota=0.4, psi=0.659, phase=1.3, geocent_time=1.2, ra=1.3, dec=-1.2)

        ifos = bilby.gw.detector.InterferometerList(['H1'])
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=self.duration)

        self.priors = bilby.gw.prior.BBHPriorDict()
        self.priors['geocent_time'] = bilby.core.prior.Uniform(1.1, 1.3)

        non_roq_wfg = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=20.0, minimum_frequency=20.0,
                approximant='IMRPhenomPv2'))

        ifos.inject_signal(
            parameters=self.test_parameters, waveform_generator=non_roq_wfg)

        roq_wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.roq,
            waveform_arguments=dict(
                frequency_nodes_linear=freq_nodes_linear,
                frequency_nodes_quadratic=freq_nodes_quadratic,
                reference_frequency=20., minimum_frequency=20.,
                approximant='IMRPhenomPv2'))

        self.non_roq_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=non_roq_wfg)

        self.roq_likelihood = bilby.gw.likelihood.ROQGravitationalWaveTransient(
            interferometers=ifos, waveform_generator=roq_wfg,
            linear_matrix=basis_matrix_linear,
            quadratic_matrix=basic_matrix_quadratic, priors=self.priors)
        pass

    def tearDown(self):
        pass

    def test_matches_non_roq(self):
        self.non_roq_likelihood.parameters.update(self.test_parameters)
        self.roq_likelihood.parameters.update(self.test_parameters)
        self.assertAlmostEqual(
            self.non_roq_likelihood.log_likelihood_ratio(),
            self.roq_likelihood.log_likelihood_ratio(), 0)

    def test_time_prior_out_of_bounds_returns_zero(self):
        self.roq_likelihood.parameters.update(self.test_parameters)
        self.roq_likelihood.parameters['geocent_time'] = -5
        self.assertEqual(
            self.roq_likelihood.log_likelihood_ratio(), np.nan_to_num(-np.inf))


class TestBBHLikelihoodSetUp(unittest.TestCase):

    def setUp(self):
        self.ifos = bilby.gw.detector.InterferometerList(['H1'])

    def tearDown(self):
        del self.ifos

    def test_instantiation(self):
        self.like = bilby.gw.likelihood.get_binary_black_hole_likelihood(
            self.ifos)


if __name__ == '__main__':
    unittest.main()
