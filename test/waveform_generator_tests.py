from __future__ import absolute_import
import unittest
import tupak
import numpy as np
import mock
from mock import MagicMock


def dummy_func_array_return_value(frequency_array, amplitude, mu, sigma, ra, dec, geocent_time, psi, **kwargs):
    return amplitude + mu + frequency_array + sigma + ra + dec + geocent_time + psi


def dummy_func_dict_return_value(frequency_array, amplitude, mu, sigma, ra, dec, geocent_time, psi, **kwargs):
    ht = {'plus': amplitude + mu + frequency_array + sigma + ra + dec + geocent_time + psi,
          'cross': amplitude + mu + frequency_array + sigma + ra + dec + geocent_time + psi}
    return ht


class TestWaveformGeneratorInstantiationWithoutOptionalParameters(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = \
            tupak.gw.waveform_generator.WaveformGenerator(1, 4096,
                                                          frequency_domain_source_model=dummy_func_dict_return_value)
        self.simulation_parameters = dict(amplitude=1e-21, mu=100, sigma=1,
                                          ra=1.375,
                                          dec=-1.2108,
                                          geocent_time=1126259642.413,
                                          psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_duration(self):
        self.assertEqual(self.waveform_generator.duration, 1)

    def test_sampling_frequency(self):
        self.assertEqual(self.waveform_generator.sampling_frequency, 4096)

    def test_source_model(self):
        self.assertEqual(self.waveform_generator.frequency_domain_source_model, dummy_func_dict_return_value)

    def test_frequency_array_type(self):
        self.assertIsInstance(self.waveform_generator.frequency_array, np.ndarray)

    def test_time_array_type(self):
        self.assertIsInstance(self.waveform_generator.time_array, np.ndarray)

    def test_source_model_parameters(self):
        self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
                             sorted(list(self.simulation_parameters.keys())))


class TestWaveformArgumentsSetting(unittest.TestCase):
    def setUp(self):
        self.waveform_generator = \
            tupak.gw.waveform_generator.WaveformGenerator(1, 4096,
                                                          frequency_domain_source_model=dummy_func_dict_return_value,
                                                          waveform_arguments=dict(test='test', arguments='arguments'))

    def tearDown(self):
        del self.waveform_generator

    def test_waveform_arguments_init_setting(self):
        self.assertDictEqual(self.waveform_generator.waveform_arguments,
                             dict(test='test', arguments='arguments'))


class TestSetters(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = \
            tupak.gw.waveform_generator.WaveformGenerator(1, 4096,
                                                          frequency_domain_source_model=dummy_func_dict_return_value)
        self.simulation_parameters = dict(amplitude=1e-21, mu=100, sigma=1,
                                          ra=1.375,
                                          dec=-1.2108,
                                          geocent_time=1126259642.413,
                                          psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameter_setter_sets_expected_values_with_expected_keys(self):
        self.waveform_generator.parameters = self.simulation_parameters
        for key in self.simulation_parameters:
            self.assertEqual(self.waveform_generator.parameters[key], self.simulation_parameters[key])

    def test_parameter_setter_none_handling(self):
        self.waveform_generator.parameters = None
        self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
                             sorted(list(self.simulation_parameters.keys())))

    def test_frequency_array_setter(self):
        new_frequency_array = np.arange(1, 100)
        self.waveform_generator.frequency_array = new_frequency_array
        self.assertTrue(np.array_equal(new_frequency_array, self.waveform_generator.frequency_array))

    def test_time_array_setter(self):
        new_time_array = np.arange(1, 100)
        self.waveform_generator.time_array = new_time_array
        self.assertTrue(np.array_equal(new_time_array, self.waveform_generator.time_array))

    def test_parameters_set_from_frequency_domain_source_model(self):
        self.waveform_generator.frequency_domain_source_model = dummy_func_dict_return_value
        self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
                             sorted(list(self.simulation_parameters.keys())))

    def test_parameters_set_from_time_domain_source_model(self):
        self.waveform_generator.time_domain_source_model = dummy_func_dict_return_value
        self.assertListEqual(sorted(list(self.waveform_generator.parameters.keys())),
                             sorted(list(self.simulation_parameters.keys())))


class TestFrequencyDomainStrainMethod(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = \
            tupak.gw.waveform_generator.WaveformGenerator(duration=1, sampling_frequency=4096,
                                                          frequency_domain_source_model=dummy_func_dict_return_value)
        self.simulation_parameters = dict(amplitude=1e-2, mu=100, sigma=1,
                                          ra=1.375,
                                          dec=-1.2108,
                                          geocent_time=1126259642.413,
                                          psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameter_conversion_is_called(self):
        self.waveform_generator.parameter_conversion = MagicMock(side_effect=KeyError('test'))
        with self.assertRaises(KeyError):
            self.waveform_generator.frequency_domain_strain()

    def test_frequency_domain_source_model_call(self):
        self.waveform_generator.parameters = self.simulation_parameters
        expected = self.waveform_generator.frequency_domain_source_model(self.waveform_generator.frequency_array,
                                                                         self.simulation_parameters['amplitude'],
                                                                         self.simulation_parameters['mu'],
                                                                         self.simulation_parameters['sigma'],
                                                                         self.simulation_parameters['ra'],
                                                                         self.simulation_parameters['dec'],
                                                                         self.simulation_parameters['geocent_time'],
                                                                         self.simulation_parameters['psi'])
        actual = self.waveform_generator.frequency_domain_strain()
        self.assertTrue(np.array_equal(expected['plus'], actual['plus']))
        self.assertTrue(np.array_equal(expected['cross'], actual['cross']))

    def test_time_domain_source_model_call_with_ndarray(self):
        self.waveform_generator.frequency_domain_source_model = None
        self.waveform_generator.time_domain_source_model = dummy_func_array_return_value
        self.waveform_generator.parameters = self.simulation_parameters

        def side_effect(value, value2):
            return value

        with mock.patch('tupak.core.utils.nfft') as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.time_domain_strain()
            actual = self.waveform_generator.frequency_domain_strain()
            self.assertTrue(np.array_equal(expected, actual))

    def test_time_domain_source_model_call_with_dict(self):
        self.waveform_generator.frequency_domain_source_model = None
        self.waveform_generator.time_domain_source_model = dummy_func_dict_return_value
        self.waveform_generator.parameters = self.simulation_parameters

        def side_effect(value, value2):
            return value, self.waveform_generator.frequency_array

        with mock.patch('tupak.core.utils.nfft') as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.time_domain_strain()
            actual = self.waveform_generator.frequency_domain_strain()
            self.assertTrue(np.array_equal(expected['plus'], actual['plus']))
            self.assertTrue(np.array_equal(expected['cross'], actual['cross']))

    def test_no_source_model_given(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = None
        with self.assertRaises(RuntimeError):
            self.waveform_generator.frequency_domain_strain()

    def test_key_popping(self):
        self.waveform_generator.parameter_conversion = MagicMock(return_value=(dict(amplitude=1e-21, mu=100, sigma=1,
                                                                                    ra=1.375, dec=-1.2108,
                                                                                    geocent_time=1126259642.413,
                                                                                    psi=2.659, c=None, d=None),
                                                                               ['c', 'd']))
        try:
            self.waveform_generator.frequency_domain_strain()
        except RuntimeError:
            pass
        self.assertListEqual(sorted(self.waveform_generator.parameters.keys()),
                             sorted(['amplitude', 'mu', 'sigma', 'ra', 'dec', 'geocent_time', 'psi']))


class TestTimeDomainStrainMethod(unittest.TestCase):

    def setUp(self):
        self.waveform_generator = \
            tupak.gw.waveform_generator.WaveformGenerator(1, 4096,
                                                          time_domain_source_model=dummy_func_dict_return_value)
        self.simulation_parameters = dict(amplitude=1e-21, mu=100, sigma=1,
                                          ra=1.375,
                                          dec=-1.2108,
                                          geocent_time=1126259642.413,
                                          psi=2.659)

    def tearDown(self):
        del self.waveform_generator
        del self.simulation_parameters

    def test_parameter_conversion_is_called(self):
        self.waveform_generator.parameter_conversion = MagicMock(side_effect=KeyError('test'))
        with self.assertRaises(KeyError):
            self.waveform_generator.time_domain_strain()

    def test_time_domain_source_model_call(self):
        self.waveform_generator.parameters = self.simulation_parameters
        expected = self.waveform_generator.time_domain_source_model(self.waveform_generator.time_array,
                                                                    self.simulation_parameters['amplitude'],
                                                                    self.simulation_parameters['mu'],
                                                                    self.simulation_parameters['sigma'],
                                                                    self.simulation_parameters['ra'],
                                                                    self.simulation_parameters['dec'],
                                                                    self.simulation_parameters['geocent_time'],
                                                                    self.simulation_parameters['psi'])
        actual = self.waveform_generator.time_domain_strain()
        self.assertTrue(np.array_equal(expected['plus'], actual['plus']))
        self.assertTrue(np.array_equal(expected['cross'], actual['cross']))

    def test_frequency_domain_source_model_call_with_ndarray(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = dummy_func_array_return_value
        self.waveform_generator.parameters = self.simulation_parameters

        def side_effect(value, value2):
            return value

        with mock.patch('tupak.core.utils.infft') as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.frequency_domain_strain()
            actual = self.waveform_generator.time_domain_strain()
            self.assertTrue(np.array_equal(expected, actual))

    def test_frequency_domain_source_model_call_with_dict(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = dummy_func_dict_return_value
        self.waveform_generator.parameters = self.simulation_parameters

        def side_effect(value, value2):
            return value

        with mock.patch('tupak.core.utils.infft') as m:
            m.side_effect = side_effect
            expected = self.waveform_generator.frequency_domain_strain()
            actual = self.waveform_generator.time_domain_strain()
            self.assertTrue(np.array_equal(expected['plus'], actual['plus']))
            self.assertTrue(np.array_equal(expected['cross'], actual['cross']))

    def test_no_source_model_given(self):
        self.waveform_generator.time_domain_source_model = None
        self.waveform_generator.frequency_domain_source_model = None
        with self.assertRaises(RuntimeError):
            self.waveform_generator.time_domain_strain()

    def test_key_popping(self):
        self.waveform_generator.parameter_conversion = MagicMock(return_value=(dict(amplitude=1e-2,
                                                                                    mu=100,
                                                                                    sigma=1,
                                                                                    ra=1.375, dec=-1.2108,
                                                                                    geocent_time=1126259642.413,
                                                                                    psi=2.659, c=None, d=None),
                                                                               ['c', 'd']))
        try:
            self.waveform_generator.time_domain_strain()
        except RuntimeError:
            pass
        self.assertListEqual(sorted(self.waveform_generator.parameters.keys()),
                             sorted(['amplitude', 'mu', 'sigma', 'ra', 'dec', 'geocent_time', 'psi']))


if __name__ == '__main__':
    unittest.main()
